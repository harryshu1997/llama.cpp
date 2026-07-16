#include "phone_pim_store.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace pp = phone_pim;

namespace {

int checks = 0;
int failures = 0;

void check(bool condition, const char * name) {
    ++checks;
    std::printf("  %s %s\n", condition ? "PASS" : "FAIL", name);
    if (!condition) {
        ++failures;
    }
}

pp::StageObjectSpec make_spec(const std::vector<uint8_t> & data, uint32_t chunk_bytes) {
    pp::StageObjectSpec spec;
    spec.bytes = data.size();
    spec.sha256 = pp::sha256(data.data(), data.size());
    spec.chunk_bytes = chunk_bytes;
    uint64_t offset = 0;
    uint32_t index = 0;
    while (offset < data.size()) {
        pp::StageChunkSpec chunk;
        chunk.index = index++;
        chunk.offset = offset;
        chunk.bytes = static_cast<uint32_t>(std::min<uint64_t>(chunk_bytes, data.size() - offset));
        chunk.sha256 = pp::sha256(data.data() + offset, chunk.bytes);
        spec.chunks.push_back(chunk);
        offset += chunk.bytes;
    }
    spec.manifest_sha256 = pp::stage_manifest_sha256(spec);
    return spec;
}

bool put(
        pp::ArtifactStore & store,
        uint64_t ticket,
        uint64_t generation,
        const pp::StageObjectSpec & spec,
        const std::vector<uint8_t> & data,
        uint32_t index,
        pp::StageProgress & progress,
        pp::StageMetrics & metrics,
        std::string & error) {
    const pp::StageChunkSpec & chunk = spec.chunks[index];
    return store.put_chunk(
            ticket, generation, index, chunk.offset, data.data() + chunk.offset, chunk.bytes,
            chunk.sha256, progress, metrics, error);
}

void remove_tree(const std::string & root) {
    DIR * dir = opendir(root.c_str());
    if (dir != nullptr) {
        while (dirent * entry = readdir(dir)) {
            if (std::strcmp(entry->d_name, ".") == 0 || std::strcmp(entry->d_name, "..") == 0) {
                continue;
            }
            const std::string path = root + "/" + entry->d_name;
            unlink(path.c_str());
        }
        closedir(dir);
    }
    rmdir(root.c_str());
}

} // namespace

int main() {
    char root_template[] = "/tmp/phone_pim_store_XXXXXX";
    char * root_value = mkdtemp(root_template);
    if (root_value == nullptr) {
        std::fprintf(stderr, "mkdtemp failed\n");
        return 1;
    }
    const std::string root = root_value;

    pp::StoreLimits limits;
    limits.max_store_bytes = 1024 * 1024;
    limits.max_object_bytes = 512 * 1024;
    limits.min_free_bytes = 0;
    limits.max_chunks = 32;
    limits.max_chunk_bytes = 64;

    pp::ArtifactStore store;
    std::string error;
    check(store.open(root, limits, error), "private durable store opens");
    pp::ArtifactStore second;
    error.clear();
    check(!second.open(root, limits, error), "second store process is excluded by the lock");

    std::vector<uint8_t> data(23);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<uint8_t>(i * 17 + 3);
    const pp::StageObjectSpec spec = make_spec(data, 7);
    error.clear();
    check(pp::validate_stage_spec(spec, limits, error), "exact odd-sized chunk map validates");
    pp::StageObjectSpec gap = spec;
    ++gap.chunks[1].offset;
    gap.manifest_sha256 = pp::stage_manifest_sha256(gap);
    error.clear();
    check(!pp::validate_stage_spec(gap, limits, error), "gapped chunk map is rejected");
    pp::StageObjectSpec relabeled = spec;
    relabeled.manifest_sha256[0] ^= 1;
    error.clear();
    check(!pp::validate_stage_spec(relabeled, limits, error), "manifest digest binds the chunk map");

    constexpr uint64_t generation = 3;
    uint64_t ticket = 101;
    pp::StageProgress progress;
    error.clear();
    check(store.begin(ticket, generation, spec, progress, error) &&
                  progress.state == pp::StageState::receiving && progress.verified_bytes == 0 &&
                  progress.next_chunk == 0,
          "new transfer begins at an empty durable prefix");
    const std::string reserved_path = root + "/" + pp::hex_sha256(spec.sha256) + ".part";
    struct stat reserved_st = {};
    check(stat(reserved_path.c_str(), &reserved_st) == 0 &&
                  static_cast<uint64_t>(reserved_st.st_size) == spec.bytes,
          "begin reserves the complete logical object before accepting chunks");

    pp::StageMetrics metrics;
    std::array<uint8_t, 32> wrong_digest = spec.chunks[0].sha256;
    wrong_digest[0] ^= 1;
    error.clear();
    check(!store.put_chunk(
                    ticket, generation, 0, 0, data.data(), spec.chunks[0].bytes,
                    wrong_digest, progress, metrics, error),
          "chunk identity differing from the manifest is rejected");
    std::vector<uint8_t> corrupt = data;
    corrupt[0] ^= 1;
    error.clear();
    check(!store.put_chunk(
                    ticket, generation, 0, 0, corrupt.data(), spec.chunks[0].bytes,
                    spec.chunks[0].sha256, progress, metrics, error),
          "payload corruption is rejected before write");
    error.clear();
    check(store.active_progress(progress, error) && progress.verified_bytes == 0,
          "rejected chunks do not advance the durable prefix");

    error.clear();
    check(put(store, ticket, generation, spec, data, 0, progress, metrics, error) &&
                  progress.verified_bytes == 7 && progress.next_chunk == 1 &&
                  progress.prefix_sha256 == pp::sha256(data.data(), 7),
          "accepted chunk is durable before acknowledgement");
    error.clear();
    check(put(store, ticket, generation, spec, data, 0, progress, metrics, error) &&
                  metrics.duplicate_chunks == 1 && progress.verified_bytes == 7 &&
                  progress.prefix_sha256 == pp::sha256(data.data(), 7),
          "exact duplicate chunk is an idempotent acknowledgement");
    error.clear();
    check(!put(store, ticket, generation, spec, data, 2, progress, metrics, error),
          "reordered chunk is rejected");
    error.clear();
    check(!store.commit(ticket, generation, spec.manifest_sha256, progress, metrics, error),
          "incomplete staged object cannot publish");
    pp::Fd unpublished;
    error.clear();
    check(!store.lookup(spec.bytes, spec.sha256, unpublished, error),
          "staging bytes are never visible through lookup");

    store.detach_active();
    ticket = 202;
    error.clear();
    check(store.begin(ticket, generation, spec, progress, error) && progress.verified_bytes == 7 &&
                  progress.next_chunk == 1 &&
                  progress.prefix_sha256 == pp::sha256(data.data(), 7),
          "reconnect rehashes and resumes the exact durable prefix");
    for (uint32_t i = 1; i < spec.chunks.size(); ++i) {
        error.clear();
        check(put(store, ticket, generation, spec, data, i, progress, metrics, error),
              "remaining ordered chunk is accepted");
    }
    error.clear();
    check(store.commit(ticket, generation, spec.manifest_sha256, progress, metrics, error) &&
                  progress.state == pp::StageState::published && progress.verified_bytes == data.size(),
          "complete verified object publishes atomically");
    pp::Fd published;
    error.clear();
    std::vector<uint8_t> readback(data.size());
    check(store.lookup(spec.bytes, spec.sha256, published, error) &&
                  pread(published.get(), readback.data(), readback.size(), 0) ==
                          static_cast<ssize_t>(readback.size()) && readback == data,
          "published lookup returns the verified bytes");
    error.clear();
    check(store.begin(303, generation, spec, progress, error) &&
                  progress.state == pp::StageState::published && progress.prefix_sha256 == spec.sha256,
          "existing verified publication is an idempotent cache hit");
    std::vector<uint8_t> active_data(15, 0x4a);
    const pp::StageObjectSpec active_spec = make_spec(active_data, 8);
    error.clear();
    const bool active_started = store.begin(1301, generation, active_spec, progress, error) &&
            put(store, 1301, generation, active_spec, active_data, 0, progress, metrics, error);
    pp::StageProgress still_active;
    error.clear();
    const bool cache_hit_rejected = !store.begin(1302, generation, spec, progress, error);
    error.clear();
    check(active_started && cache_hit_rejected && store.active_progress(still_active, error) &&
                  still_active.ticket_id == 1301 &&
                  still_active.manifest_sha256 == active_spec.manifest_sha256,
          "published cache hit cannot discard a different active transfer");
    error.clear();
    check(store.abort(1301, generation, false, error),
          "active transfer remains abortable after rejected cache hit");
    const std::string published_path = root + "/" + pp::hex_sha256(spec.sha256) + ".gguf";
    errno = 0;
    pp::Fd changed_file(open(published_path.c_str(), O_WRONLY | O_CLOEXEC));
    const int writable_error = errno;
    check(!changed_file.valid() && (writable_error == EACCES || writable_error == EPERM),
          "published object is read-only after durable publication");
    sleep(1);
    const bool made_writable = chmod(published_path.c_str(), 0600) == 0;
    changed_file.reset(open(published_path.c_str(), O_WRONLY | O_CLOEXEC));
    const uint8_t changed_byte = static_cast<uint8_t>(data[0] ^ 0xff);
    const bool changed_after_cache = made_writable && changed_file.valid() &&
            pwrite(changed_file.get(), &changed_byte, 1, 0) == 1 &&
            fdatasync(changed_file.get()) == 0 && fchmod(changed_file.get(), 0400) == 0;
    changed_file.reset();
    pp::Fd changed_lookup;
    error.clear();
    check(changed_after_cache && !store.lookup(spec.bytes, spec.sha256, changed_lookup, error),
          "verified descriptor cache fails closed when the published inode changes");
    error.clear();
    check(store.begin(304, generation, spec, progress, error) &&
                  progress.state == pp::StageState::receiving && progress.verified_bytes == 0,
          "changed cached publication is quarantined before replacement");
    error.clear();
    check(store.abort(304, generation, false, error),
          "replacement after cache invalidation can be aborted");

    std::vector<uint8_t> bad_full_data(19, 0x5a);
    pp::StageObjectSpec bad_full = make_spec(bad_full_data, 8);
    const char wrong_identity[] = "wrong-full-object-identity";
    bad_full.sha256 = pp::sha256(wrong_identity, sizeof(wrong_identity) - 1);
    bad_full.manifest_sha256 = pp::stage_manifest_sha256(bad_full);
    ticket = 404;
    error.clear();
    check(store.begin(ticket, generation, bad_full, progress, error),
          "full-hash mismatch case begins with valid chunk descriptors");
    for (uint32_t i = 0; i < bad_full.chunks.size(); ++i) {
        error.clear();
        check(put(store, ticket, generation, bad_full, bad_full_data, i, progress, metrics, error),
              "full-hash mismatch case accepts individually valid chunks");
    }
    error.clear();
    check(!store.commit(ticket, generation, bad_full.manifest_sha256, progress, metrics, error),
          "full-file hash mismatch quarantines instead of publishing");
    pp::Fd bad_lookup;
    error.clear();
    check(!store.lookup(bad_full.bytes, bad_full.sha256, bad_lookup, error),
          "quarantined object is not visible");

    std::vector<uint8_t> recovery_data(18, 0x31);
    const pp::StageObjectSpec recovery = make_spec(recovery_data, 6);
    ticket = 505;
    error.clear();
    check(store.begin(ticket, generation, recovery, progress, error) &&
                  put(store, ticket, generation, recovery, recovery_data, 0, progress, metrics, error),
          "recovery case writes one durable chunk");
    store.detach_active();
    const std::string partial_path = root + "/" + pp::hex_sha256(recovery.sha256) + ".part";
    pp::Fd partial(open(partial_path.c_str(), O_WRONLY));
    const uint8_t changed = 0xff;
    const bool mutated = partial.valid() && pwrite(partial.get(), &changed, 1, 0) == 1 &&
                         fdatasync(partial.get()) == 0;
    partial.reset();
    ticket = 606;
    error.clear();
    check(mutated && store.begin(ticket, generation, recovery, progress, error) &&
                  progress.verified_bytes == 0 && progress.next_chunk == 0,
          "corrupt recovered prefix is truncated to the last verified boundary");
    error.clear();
    check(store.abort(ticket, generation, false, error), "abort removes the active staging object");

    std::vector<uint8_t> symlink_data(9, 0x77);
    const pp::StageObjectSpec symlink_spec = make_spec(symlink_data, 5);
    const std::string symlink_path = root + "/" + pp::hex_sha256(symlink_spec.sha256) + ".part";
    const bool linked = symlink("/etc/passwd", symlink_path.c_str()) == 0;
    error.clear();
    check(linked && !store.begin(707, generation, symlink_spec, progress, error),
          "symlink staging entry is rejected");
    unlink(symlink_path.c_str());

    std::vector<uint8_t> hardlink_data(11, 0x42);
    const pp::StageObjectSpec hardlink_spec = make_spec(hardlink_data, 6);
    char external_template[] = "/tmp/phone_pim_external_XXXXXX";
    pp::Fd external(mkstemp(external_template));
    const std::string hardlink_path =
            root + "/" + pp::hex_sha256(hardlink_spec.sha256) + ".part";
    const std::array<uint8_t, 5> sentinel = {9, 8, 7, 6, 5};
    const bool external_file_ready = external.valid() &&
            write(external.get(), sentinel.data(), sentinel.size()) ==
                    static_cast<ssize_t>(sentinel.size()) &&
            fdatasync(external.get()) == 0;
    errno = 0;
    const int hardlink_rc = external_file_ready
            ? link(external_template, hardlink_path.c_str()) : -1;
    const int hardlink_error = errno;
    const bool hardlink_blocked = external_file_ready && hardlink_rc != 0 &&
            (hardlink_error == EPERM || hardlink_error == EACCES || hardlink_error == EOPNOTSUPP);
    error.clear();
    check(hardlink_blocked ||
                  (hardlink_rc == 0 && !store.begin(808, generation, hardlink_spec, progress, error)),
          "writable staging hard links are blocked by the platform or rejected by the store");
    std::array<uint8_t, 5> sentinel_read = {};
    check(pread(external.get(), sentinel_read.data(), sentinel_read.size(), 0) ==
                  static_cast<ssize_t>(sentinel_read.size()) && sentinel_read == sentinel,
          "rejected staging hard link does not mutate its external inode");
    unlink(hardlink_path.c_str());
    external.reset();
    unlink(external_template);

    const std::string fifo_path = root + "/unexpected";
    errno = 0;
    const bool fifo_created = mkfifo(fifo_path.c_str(), 0600) == 0;
    const int fifo_error = errno;
    const bool fifo_blocked = !fifo_created &&
            (fifo_error == EPERM || fifo_error == EACCES || fifo_error == EOPNOTSUPP);
    std::vector<uint8_t> fifo_data(10, 0x19);
    const pp::StageObjectSpec fifo_spec = make_spec(fifo_data, 5);
    error.clear();
    check(fifo_blocked ||
                  (fifo_created && !store.begin(909, generation, fifo_spec, progress, error)),
          "unknown FIFOs are blocked by the platform or fail closed in the store scan");
    unlink(fifo_path.c_str());
    unlink((root + "/" + pp::hex_sha256(fifo_spec.sha256) + ".part").c_str());

    std::vector<uint8_t> corrupt_final_data(13, 0x26);
    const pp::StageObjectSpec corrupt_final = make_spec(corrupt_final_data, 7);
    const std::string corrupt_final_path =
            root + "/" + pp::hex_sha256(corrupt_final.sha256) + ".gguf";
    pp::Fd corrupt_file(open(
            corrupt_final_path.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600));
    std::vector<uint8_t> wrong_final(corrupt_final_data.size(), 0x27);
    const bool corrupt_ready = corrupt_file.valid() &&
            write(corrupt_file.get(), wrong_final.data(), wrong_final.size()) ==
                    static_cast<ssize_t>(wrong_final.size()) && fdatasync(corrupt_file.get()) == 0;
    corrupt_file.reset();
    error.clear();
    check(corrupt_ready && store.begin(1001, generation, corrupt_final, progress, error) &&
                  progress.state == pp::StageState::receiving && progress.verified_bytes == 0,
          "corrupt published bytes are quarantined before a new transfer begins");
    error.clear();
    check(store.abort(1001, generation, false, error),
          "replacement transfer after corrupt publication can be aborted cleanly");

    char crash_root_template[] = "/tmp/phone_pim_crash_XXXXXX";
    char * crash_root_value = mkdtemp(crash_root_template);
    std::vector<uint8_t> crash_data(21, 0x6d);
    const pp::StageObjectSpec crash_spec = make_spec(crash_data, 7);
    const pid_t child = crash_root_value == nullptr ? -1 : fork();
    if (child == 0) {
        pp::ArtifactStore child_store;
        pp::StageProgress child_progress;
        pp::StageMetrics child_metrics;
        std::string child_error;
        const bool ok = child_store.open(crash_root_value, limits, child_error) &&
                child_store.begin(1101, generation, crash_spec, child_progress, child_error) &&
                put(child_store, 1101, generation, crash_spec, crash_data, 0,
                    child_progress, child_metrics, child_error);
        _exit(ok ? 0 : 2);
    }
    int child_status = 0;
    const bool child_ok = child > 0 && waitpid(child, &child_status, 0) == child &&
            WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0;
    pp::ArtifactStore recovered_store;
    pp::StageProgress recovered_progress;
    error.clear();
    const bool recovered = child_ok && recovered_store.open(crash_root_value, limits, error) &&
            recovered_store.begin(1102, generation, crash_spec, recovered_progress, error);
    check(recovered && recovered_progress.verified_bytes == 7 &&
                  recovered_progress.prefix_sha256 == pp::sha256(crash_data.data(), 7),
          "a new process resumes the exact fdatasync-acknowledged prefix");
    if (recovered) {
        error.clear();
        recovered_store.abort(1102, generation, false, error);
    }
    recovered_store.detach_active();
    if (crash_root_value != nullptr) remove_tree(crash_root_value);

    char pair_root_template[] = "/tmp/phone_pim_pair_XXXXXX";
    char * pair_root_value = mkdtemp(pair_root_template);
    std::vector<uint8_t> pair_data(17, 0x54);
    const pp::StageObjectSpec pair_spec = make_spec(pair_data, 9);
    const std::string pair_base = pair_root_value == nullptr ? "" :
            std::string(pair_root_value) + "/" + pp::hex_sha256(pair_spec.sha256);
    pp::Fd pair_part(pair_base.empty() ? -1 : open(
            (pair_base + ".part").c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600));
    const bool pair_file_ready = pair_part.valid() &&
            write(pair_part.get(), pair_data.data(), pair_data.size()) ==
                    static_cast<ssize_t>(pair_data.size()) && fsync(pair_part.get()) == 0;
    errno = 0;
    const int pair_link_rc = pair_file_ready
            ? link((pair_base + ".part").c_str(), (pair_base + ".gguf").c_str()) : -1;
    const int pair_link_error = errno;
    const bool pair_hardlink_blocked = pair_file_ready && pair_link_rc != 0 &&
            (pair_link_error == EPERM || pair_link_error == EACCES || pair_link_error == EOPNOTSUPP);
    const bool pair_ready = pair_file_ready && pair_link_rc == 0;
    pair_part.reset();
    pp::ArtifactStore pair_store;
    pp::StageProgress pair_progress;
    error.clear();
    const bool pair_recovered = pair_ready && pair_store.open(pair_root_value, limits, error) &&
            pair_store.begin(1201, generation, pair_spec, pair_progress, error);
    struct stat pair_st = {};
    check(pair_hardlink_blocked ||
                  (pair_recovered && pair_progress.state == pp::StageState::published &&
                   lstat((pair_base + ".part").c_str(), &pair_st) != 0 && errno == ENOENT &&
                   stat((pair_base + ".gguf").c_str(), &pair_st) == 0 && pair_st.st_nlink == 1),
          "legacy hard-link publish pairs are blocked or reconciled at store open");
    pair_store.detach_active();
    if (pair_root_value != nullptr) remove_tree(pair_root_value);

    store.detach_active();
    remove_tree(root);
    std::printf("store tests: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
