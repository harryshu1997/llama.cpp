"""Private worker variants; canonical worker and default mode stay unchanged."""

from build_pixel_cpu_gpu import replace_once


JOIN = r'''
host_tensor pixel_join(const host_tensor & a, const host_tensor & b, bool rows) {
    if (a.type != b.type || (rows ? a.ne0 != b.ne0 : a.ne1 != b.ne1)) std::abort();
    host_tensor result;
    result.type = a.type;
    result.ne0 = rows ? a.ne0 : a.ne0 + b.ne0;
    result.ne1 = rows ? a.ne1 + b.ne1 : a.ne1;
    result.bytes.resize(a.bytes.size() + b.bytes.size());
    if (rows) {
        memcpy(result.bytes.data(), a.bytes.data(), a.bytes.size());
        memcpy(result.bytes.data() + a.bytes.size(), b.bytes.data(), b.bytes.size());
    } else {
        const size_t na = ggml_row_size(a.type, a.ne0);
        const size_t nb = ggml_row_size(b.type, b.ne0);
        for (int64_t row = 0; row < a.ne1; ++row) {
            memcpy(result.bytes.data() + row * (na + nb), a.bytes.data() + row * na, na);
            memcpy(result.bytes.data() + row * (na + nb) + na, b.bytes.data() + row * nb, nb);
        }
    }
    return result;
}
'''


def transform(source):
    source = replace_once(source, "struct layer_state {", JOIN + "\nstruct layer_state {")
    source = replace_once(source, "    std::vector<block> blocks;", r'''
    std::vector<block> blocks;
    bool coalesced = false;
    std::pair<size_t, size_t> selection(int64_t columns) const {
        if (coalesced && columns == 17408) return {0, 1};
        size_t first = blocks.size();
        int64_t selected = 0;
        while (first > 0 && selected < columns) selected += blocks[--first].columns;
        return selected == columns ? std::make_pair(first, blocks.size()) : std::make_pair(size_t(0), size_t(0));
    }''')
    source = replace_once(source, "    const bool dual = secondary.enabled();", r'''
    const bool dual = secondary.enabled();
    const bool pixel_coalesce = getenv("S42_PIXEL_COALESCE_FULL") != nullptr;
    const bool pixel_packed = getenv("S42_PIXEL_PACKED_WEIGHTS") != nullptr;
    if (pixel_coalesce && (!dual || !secondary.cpu_half_columns)) {
        fprintf(stderr, "[ffn-worker] coalescing requires a fine CPU/GPU split\n");
        return 2;
    }
    if (pixel_packed && secondary.cpu_half_columns % 256 != 0) {
        fprintf(stderr, "[ffn-worker] packed ratios require 256-column alignment\n");
        return 2;
    }''')
    source = replace_once(source, "            gate_full.type != up_full.type || gate_full.type != down_full.type)",
                          "            (!pixel_packed && (gate_full.type != up_full.type || gate_full.type != down_full.type)))")
    source = replace_once(source, "             strcasecmp(ggml_type_name(gate_full.type), shard.weight_type.c_str()) != 0))", 
                          "             (!pixel_packed && strcasecmp(ggml_type_name(gate_full.type), shard.weight_type.c_str()) != 0)))")
    source = replace_once(source, "        if (states.empty()) {", r'''
        if (pixel_packed) {
            for (const host_tensor * tensor : {&gate_full, &up_full, &down_full}) {
                if (tensor->type != GGML_TYPE_Q4_K && tensor->type != GGML_TYPE_Q6_K) {
                    fprintf(stderr, "[ffn-worker] packed experiment requires original Q4_K/Q6_K weights\n");
                    return 1;
                }
            }
            fprintf(stderr, "S42PIXELPACKED layer=%d gate=%s up=%s down=%s\n", layer,
                    ggml_type_name(gate_full.type), ggml_type_name(up_full.type), ggml_type_name(down_full.type));
        }
        if (states.empty()) {''')
    source = replace_once(source, "    if (ggml_is_quantized(weight_type)) {",
                          "    if (ggml_is_quantized(weight_type) && !pixel_packed) {")
    source = replace_once(source, "        states.push_back(std::move(state));", r'''
        if (pixel_coalesce) {
            layer_state::block full;
            full.offset = offset;
            full.columns = cfg.columns;
            full.secondary_columns = 2 * (8704 - secondary.cpu_half_columns);
            full.gate = pixel_join(state.blocks[0].gate, state.blocks[3].gate, true);
            full.up = pixel_join(state.blocks[0].up, state.blocks[3].up, true);
            full.down = pixel_join(state.blocks[0].down, state.blocks[3].down, false);
            full.secondary_gate = pixel_join(state.blocks[1].secondary_gate, state.blocks[2].secondary_gate, true);
            full.secondary_up = pixel_join(state.blocks[1].secondary_up, state.blocks[2].secondary_up, true);
            full.secondary_down = pixel_join(state.blocks[1].secondary_down, state.blocks[2].secondary_down, false);
            std::vector<layer_state::block> packed;
            packed.push_back(std::move(full));
            packed.push_back(std::move(state.blocks[2]));
            packed.push_back(std::move(state.blocks[3]));
            state.blocks = std::move(packed);
            state.coalesced = true;
        }
        states.push_back(std::move(state));''')
    # Allocate each projection using its own GGUF type, including mixed Q4_K/Q6_K.
    for name in ("gate", "up", "down", "copy_gate", "copy_up", "copy_down", "secondary_gate", "secondary_up", "secondary_down"):
        host = name.replace("copy_", "secondary_")
        needle = f"block.{name}_weight = ggml_new_tensor_2d("
        start = source.index(needle)
        end = source.index(";", start)
        part = source[start:end]
        if part.count("weight_type") != 1:
            raise ValueError("weight allocation differs")
        source = source[:start] + part.replace("weight_type", f"block.{host}.type") + source[end:]
    selection = '''        size_t first_block = state.blocks.size();
        int64_t selected_columns = 0;
        while (first_block > 0 && selected_columns < columns) {
            --first_block;
            selected_columns += state.blocks[first_block].columns;
        }
        if (selected_columns != columns) {
            return false;
        }'''
    if source.count(selection) != 2:
        raise ValueError("graph selection differs")
    source = source.replace(selection, '''        const auto range = state.selection(columns);
        const size_t first_block = range.first;
        if (range.first == range.second) return false;''')
    source = source.replace("for (size_t index = first_block; index < state.blocks.size(); ++index)",
                            "for (size_t index = first_block; index < range.second; ++index)")
    source = replace_once(source, "        for (const layer_state::block & block : states.front().blocks) {\n            secondary_block_columns += block.secondary_columns;\n        }",
                          "        const auto full_range = states.front().selection(cfg.columns);\n"
                          "        for (size_t i = full_range.first; i < full_range.second; ++i) {\n"
                          "            secondary_block_columns += states.front().blocks[i].secondary_columns;\n        }")
    source = replace_once(source, '''        int64_t selected_columns = 0;
        for (size_t index = state.blocks.size();
             index > 0 && selected_columns < columns; --index) {
            const layer_state::block & block = state.blocks[index - 1];
            selected_columns += block.columns;''', '''        const auto range = state.selection(columns);
        for (size_t index = range.first; index < range.second; ++index) {
            const layer_state::block & block = state.blocks[index];''')
    source = replace_once(source, '    print_residency_phase(cfg, "WEIGHT_UPLOAD_READY");', r'''
    if (pixel_coalesce) fprintf(stderr, "S42PIXELCOALESCE full_gemv=6 half_gemv=6 resident_factor=1.5\n");
    print_residency_phase(cfg, "WEIGHT_UPLOAD_READY");''')
    return source
