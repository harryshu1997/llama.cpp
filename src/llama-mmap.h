#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <cstdio>

struct llama_file;
struct llama_mmap;
struct llama_mlock;

using llama_files  = std::vector<std::unique_ptr<llama_file>>;
using llama_mmaps  = std::vector<std::unique_ptr<llama_mmap>>;
using llama_mlocks = std::vector<std::unique_ptr<llama_mlock>>;

struct llama_file {
    llama_file(const char * fname, const char * mode, bool use_direct_io = false);
    llama_file(FILE * file);
    ~llama_file();

    size_t tell() const;
    size_t size() const;

    int file_id() const; // fileno overload

    void seek(size_t offset, int whence) const;

    void read_raw(void * ptr, size_t len);
    void read_raw_unsafe(void * ptr, size_t len);
    void read_aligned_chunk(void * dest, size_t size);
    uint32_t read_u32();

    void write_raw(const void * ptr, size_t len) const;
    void write_u32(uint32_t val) const;

    size_t read_alignment() const;
    bool has_direct_io() const;
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_mmap {
    llama_mmap(const llama_mmap &) = delete;
    llama_mmap(struct llama_file * file, size_t prefetch = (size_t) -1, bool numa = false);
    ~llama_mmap();

    size_t size() const;
    void * addr() const;

    void unmap_fragment(size_t first, size_t last);

    // Advise the kernel to populate the still-mapped pages covering [first, last).
    // Used when the mapping was created without a whole-file prefetch so that only the
    // retained tensor ranges are read ahead. No-op where unsupported.
    void prefetch_fragment(size_t first, size_t last);

    // Bytes of the file that are still mapped (after unmap_fragment calls).
    size_t mapped_bytes() const;

    // Page residency control for still-mapped byte ranges (S42 dormant host share).
    // release_fragments removes this process's page-table entries for the [first, last)
    // ranges (MADV_DONTNEED; ranges are aligned inward, so a page that also holds retained
    // bytes is never touched). drop_cache also advises away the now-unmapped clean pages
    // inside the cover ranges; otherwise they remain reclaimable in the file cache.
    // A later access faults the pages back in from the file. populate_fragments reads the
    // mapped pages synchronously and throws on population failure. Release returns bytes
    // advised; populate returns bytes synchronously populated, not a long-lived reservation.
    size_t release_fragments(const std::vector<std::pair<size_t, size_t>> & ranges,
                             const std::vector<std::pair<size_t, size_t>> & cache_drop_ranges,
                             bool drop_cache = true);
    size_t populate_fragments(const std::vector<std::pair<size_t, size_t>> & ranges);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_mlock {
    llama_mlock();
    ~llama_mlock();

    void init(void * ptr);
    void grow_to(size_t target_size);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

size_t llama_path_max();
