#include "phone_pim_store.h"

extern "C" {
#include "sha256.h"
}

#include <cerrno>
#include <chrono>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <limits>
#include <linux/fs.h>
#include <map>
#include <set>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <algorithm>

namespace phone_pim {
namespace {

uint64_t now_us() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

bool sync_fd(int fd, bool data_only, std::string & error) {
    int rc = 0;
    do {
        rc = data_only ? fdatasync(fd) : fsync(fd);
    } while (rc != 0 && errno == EINTR);
    if (rc != 0) {
        error = std::string(data_only ? "fdatasync failed: " : "fsync failed: ") + std::strerror(errno);
        return false;
    }
    return true;
}

bool make_read_only(int fd, std::string & error) {
    int rc = 0;
    do {
        rc = fchmod(fd, 0400);
    } while (rc != 0 && errno == EINTR);
    if (rc != 0) {
        error = "cannot make published object read-only: " + std::string(std::strerror(errno));
        return false;
    }
    return true;
}

bool truncate_fd(int fd, uint64_t size, std::string & error) {
    if (size > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
        error = "truncate size exceeds off_t";
        return false;
    }
    int rc = 0;
    do {
        rc = ftruncate(fd, static_cast<off_t>(size));
    } while (rc != 0 && errno == EINTR);
    if (rc != 0) {
        error = "ftruncate failed: " + std::string(std::strerror(errno));
        return false;
    }
    return true;
}

bool reserve_fd(int fd, uint64_t size, std::string & error) {
    if (size > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
        error = "reservation size exceeds off_t";
        return false;
    }
    int rc = 0;
    do {
        rc = posix_fallocate(fd, 0, static_cast<off_t>(size));
    } while (rc == EINTR);
    if (rc != 0) {
        error = "cannot reserve staged object storage: " + std::string(std::strerror(rc));
        return false;
    }
    return true;
}

bool rename_noreplace(
        int old_directory,
        const char * old_name,
        int new_directory,
        const char * new_name,
        std::string & error) {
#if defined(SYS_renameat2)
    int rc = 0;
    do {
        rc = static_cast<int>(syscall(
                SYS_renameat2, old_directory, old_name,
                new_directory, new_name, RENAME_NOREPLACE));
    } while (rc != 0 && errno == EINTR);
    if (rc == 0) return true;
    error = "atomic no-replace rename failed: " + std::string(std::strerror(errno));
    return false;
#else
    (void) old_directory;
    (void) old_name;
    (void) new_directory;
    (void) new_name;
    errno = ENOSYS;
    error = "atomic no-replace rename is unavailable";
    return false;
#endif
}

bool write_all_at(int fd, const uint8_t * data, size_t size, uint64_t offset, std::string & error) {
    if (offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) ||
        size > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) - offset) {
        error = "chunk range exceeds off_t";
        return false;
    }
    size_t completed = 0;
    while (completed < size) {
        const ssize_t count = pwrite(
                fd, data + completed, size - completed, static_cast<off_t>(offset + completed));
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            error = "pwrite failed: " + std::string(std::strerror(errno));
            return false;
        }
        completed += static_cast<size_t>(count);
    }
    return true;
}

bool regular_owned_file(int fd, struct stat & st, std::string & error) {
    if (fstat(fd, &st) != 0) {
        error = "fstat failed: " + std::string(std::strerror(errno));
        return false;
    }
    if (!S_ISREG(st.st_mode) || st.st_uid != geteuid() || st.st_nlink != 1 ||
        (st.st_mode & 0077) != 0 || st.st_size < 0) {
        error = "store entry is not a private regular file";
        return false;
    }
    return true;
}

bool private_regular_stat(const struct stat & st) {
    return S_ISREG(st.st_mode) && st.st_uid == geteuid() &&
           (st.st_mode & 0077) == 0 && st.st_size >= 0;
}

bool digest_name(const char * name, const char * suffix) {
    const size_t suffix_size = std::strlen(suffix);
    const size_t size = std::strlen(name);
    if (size != 64 + suffix_size || std::strcmp(name + 64, suffix) != 0) {
        return false;
    }
    for (size_t i = 0; i < 64; ++i) {
        const char value = name[i];
        if (!((value >= '0' && value <= '9') || (value >= 'a' && value <= 'f'))) {
            return false;
        }
    }
    return true;
}

bool quarantine_file_name(const char * name) {
    if (std::strlen(name) <= 69 || std::strncmp(name + 64, ".bad.", 5) != 0) {
        return false;
    }
    for (size_t i = 0; i < 64; ++i) {
        const char value = name[i];
        if (!((value >= '0' && value <= '9') || (value >= 'a' && value <= 'f'))) {
            return false;
        }
    }
    for (const char * cursor = name + 69; *cursor != '\0'; ++cursor) {
        if (*cursor < '0' || *cursor > '9') return false;
    }
    return true;
}

std::string part_name(const std::array<uint8_t, 32> & digest) {
    return hex_sha256(digest) + ".part";
}

std::string final_name(const std::array<uint8_t, 32> & digest) {
    return hex_sha256(digest) + ".gguf";
}

std::string quarantine_name(const std::array<uint8_t, 32> & digest, uint64_t ticket_id) {
    return hex_sha256(digest) + ".bad." + std::to_string(ticket_id);
}

bool same_spec(const StageObjectSpec & a, const StageObjectSpec & b) {
    if (a.bytes != b.bytes || a.sha256 != b.sha256 || a.chunk_bytes != b.chunk_bytes ||
        a.manifest_sha256 != b.manifest_sha256 || a.chunks.size() != b.chunks.size()) {
        return false;
    }
    for (size_t i = 0; i < a.chunks.size(); ++i) {
        const StageChunkSpec & x = a.chunks[i];
        const StageChunkSpec & y = b.chunks[i];
        if (x.index != y.index || x.offset != y.offset || x.bytes != y.bytes || x.sha256 != y.sha256) {
            return false;
        }
    }
    return true;
}

} // namespace

std::array<uint8_t, 32> stage_manifest_sha256(const StageObjectSpec & spec) {
    Writer writer;
    writer.string("phone-pim-stage-manifest-v1");
    writer.u64(spec.bytes);
    writer.bytes(spec.sha256.data(), spec.sha256.size());
    writer.u32(spec.chunk_bytes);
    writer.u32(static_cast<uint32_t>(spec.chunks.size()));
    for (const StageChunkSpec & chunk : spec.chunks) {
        writer.u32(chunk.index);
        writer.u64(chunk.offset);
        writer.u32(chunk.bytes);
        writer.bytes(chunk.sha256.data(), chunk.sha256.size());
    }
    return sha256(writer.data().data(), writer.data().size());
}

bool validate_stage_spec(const StageObjectSpec & spec, const StoreLimits & limits, std::string & error) {
    if (spec.bytes == 0 || spec.bytes > limits.max_object_bytes ||
        spec.bytes > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
        error = "staged object size is outside the configured bound";
        return false;
    }
    if (spec.chunk_bytes == 0 || spec.chunk_bytes > limits.max_chunk_bytes ||
        spec.chunks.empty() || spec.chunks.size() > limits.max_chunks) {
        error = "invalid staged chunk layout bounds";
        return false;
    }
    uint64_t cursor = 0;
    for (size_t i = 0; i < spec.chunks.size(); ++i) {
        const StageChunkSpec & chunk = spec.chunks[i];
        if (chunk.index != i || chunk.offset != cursor || chunk.bytes == 0 ||
            chunk.bytes > spec.chunk_bytes || chunk.bytes > limits.max_chunk_bytes ||
            (i + 1 < spec.chunks.size() && chunk.bytes != spec.chunk_bytes) ||
            cursor > spec.bytes || chunk.bytes > spec.bytes - cursor) {
            error = "chunk descriptors do not exactly tile the staged object";
            return false;
        }
        cursor += chunk.bytes;
    }
    if (cursor != spec.bytes || stage_manifest_sha256(spec) != spec.manifest_sha256) {
        error = "staged object manifest digest or coverage mismatch";
        return false;
    }
    return true;
}

struct ArtifactStore::Impl {
    struct VerifiedFile {
        Fd file;
        struct stat snapshot = {};
        uint64_t bytes = 0;
    };

    StoreLimits limits;
    std::string root;
    Fd directory;
    Fd lock;
    Fd active_file;
    StageObjectSpec active_spec;
    uint64_t active_ticket = 0;
    uint64_t active_generation = 0;
    uint32_t next_chunk = 0;
    uint64_t verified_bytes = 0;
    StageMetrics metrics;
    sha256_t prefix_hash = {};
    bool prefix_hash_valid = false;
    std::set<std::array<uint8_t, 32>> indeterminate_publications;
    std::map<std::array<uint8_t, 32>, VerifiedFile> verified_files;

    bool sync_directory(std::string & error, uint64_t * elapsed = nullptr) const {
        const uint64_t start = now_us();
        const bool ok = sync_fd(directory.get(), false, error);
        if (elapsed != nullptr) {
            *elapsed += now_us() - start;
        }
        return ok;
    }

    static bool same_snapshot(const struct stat & a, const struct stat & b) {
        return a.st_dev == b.st_dev && a.st_ino == b.st_ino &&
               a.st_size == b.st_size && a.st_mtim.tv_sec == b.st_mtim.tv_sec &&
               a.st_mtim.tv_nsec == b.st_mtim.tv_nsec &&
               a.st_ctim.tv_sec == b.st_ctim.tv_sec && a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
    }

    bool remember_verified(
            const std::array<uint8_t, 32> & digest,
            uint64_t bytes,
            std::string & error) {
        const std::string name = final_name(digest);
        Fd file(openat(directory.get(), name.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
        struct stat st = {};
        if (!file.valid() || !regular_owned_file(file.get(), st, error) ||
            static_cast<uint64_t>(st.st_size) != bytes) {
            if (error.empty()) error = "cannot cache the verified published descriptor";
            return false;
        }
        if ((st.st_mode & 0200) != 0) {
            if (!make_read_only(file.get(), error) || !sync_fd(file.get(), false, error) ||
                fstat(file.get(), &st) != 0) {
                if (error.empty()) {
                    error = "cannot seal the verified published descriptor";
                }
                return false;
            }
        }
        VerifiedFile verified;
        verified.file = std::move(file);
        verified.snapshot = st;
        verified.bytes = bytes;
        verified_files[digest] = std::move(verified);
        return true;
    }

    bool lookup_cached(
            uint64_t bytes,
            const std::array<uint8_t, 32> & digest,
            Fd & result,
            std::string & error) {
        auto cached = verified_files.find(digest);
        if (cached == verified_files.end() || cached->second.bytes != bytes) {
            return false;
        }
        const std::string name = final_name(digest);
        Fd current(openat(directory.get(), name.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
        struct stat st = {};
        if (!current.valid() || !regular_owned_file(current.get(), st, error) ||
            static_cast<uint64_t>(st.st_size) != bytes ||
            !same_snapshot(st, cached->second.snapshot)) {
            verified_files.erase(cached);
            if (error.empty()) error = "published object changed after verification";
            return false;
        }
        result = std::move(current);
        return true;
    }

    bool reconcile_publish_pair(
            const std::array<uint8_t, 32> & digest,
            std::string & error) const {
        const std::string part = part_name(digest);
        const std::string final = final_name(digest);
        struct stat part_st = {};
        struct stat final_st = {};
        const bool has_part = fstatat(
                directory.get(), part.c_str(), &part_st, AT_SYMLINK_NOFOLLOW) == 0;
        if (!has_part && errno != ENOENT) {
            error = "cannot inspect staging entry: " + std::string(std::strerror(errno));
            return false;
        }
        const bool has_final = fstatat(
                directory.get(), final.c_str(), &final_st, AT_SYMLINK_NOFOLLOW) == 0;
        if (!has_final && errno != ENOENT) {
            error = "cannot inspect published entry: " + std::string(std::strerror(errno));
            return false;
        }
        if (!has_part && !has_final) return true;
        if ((has_part && !private_regular_stat(part_st)) ||
            (has_final && !private_regular_stat(final_st))) {
            error = "store entry is not a private regular file";
            return false;
        }
        if (has_part && has_final && part_st.st_dev == final_st.st_dev &&
            part_st.st_ino == final_st.st_ino) {
            if (part_st.st_nlink != 2 || final_st.st_nlink != 2) {
                error = "published pair has an unexpected external hard link";
                return false;
            }
            if (unlinkat(directory.get(), part.c_str(), 0) != 0) {
                error = "cannot reconcile published staging link: " +
                        std::string(std::strerror(errno));
                return false;
            }
            return sync_directory(error);
        }
        if ((has_part && part_st.st_nlink != 1) || (has_final && final_st.st_nlink != 1)) {
            error = "store entry has an external hard link";
            return false;
        }
        return true;
    }

    bool reconcile_publish_pairs(std::string & error) const {
        const int duplicate = openat(
                directory.get(), ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
        if (duplicate < 0) {
            error = "cannot scan store directory: " + std::string(std::strerror(errno));
            return false;
        }
        DIR * dir = fdopendir(duplicate);
        if (dir == nullptr) {
            close(duplicate);
            error = "cannot scan store directory: " + std::string(std::strerror(errno));
            return false;
        }
        std::vector<std::array<uint8_t, 32>> digests;
        errno = 0;
        while (dirent * entry = readdir(dir)) {
            if (digest_name(entry->d_name, ".part")) {
                std::array<uint8_t, 32> digest = {};
                for (size_t i = 0; i < digest.size(); ++i) {
                    const auto nibble = [](char value) -> uint8_t {
                        return value <= '9' ? static_cast<uint8_t>(value - '0')
                                            : static_cast<uint8_t>(value - 'a' + 10);
                    };
                    digest[i] = static_cast<uint8_t>(
                            (nibble(entry->d_name[2 * i]) << 4) |
                            nibble(entry->d_name[2 * i + 1]));
                }
                digests.push_back(digest);
            }
            errno = 0;
        }
        const int scan_error = errno;
        closedir(dir);
        if (scan_error != 0) {
            error = "store directory scan failed: " + std::string(std::strerror(scan_error));
            return false;
        }
        for (const auto & digest : digests) {
            if (!reconcile_publish_pair(digest, error)) return false;
        }
        return true;
    }

    void clear_active() {
        active_file.reset();
        active_spec = {};
        active_ticket = 0;
        active_generation = 0;
        next_chunk = 0;
        verified_bytes = 0;
        metrics = {};
        prefix_hash = {};
        prefix_hash_valid = false;
    }

    bool rebuild_prefix_hash(std::string & error) {
        sha256_init(&prefix_hash);
        std::array<uint8_t, 1024 * 1024> buffer = {};
        uint64_t offset = 0;
        while (offset < verified_bytes) {
            const size_t amount = static_cast<size_t>(
                    std::min<uint64_t>(buffer.size(), verified_bytes - offset));
            size_t completed = 0;
            while (completed < amount) {
                const ssize_t count = pread(
                        active_file.get(), buffer.data() + completed, amount - completed,
                        static_cast<off_t>(offset + completed));
                if (count < 0 && errno == EINTR) continue;
                if (count <= 0) {
                    error = count == 0 ? "short read rebuilding durable prefix"
                                       : "prefix read failed: " + std::string(std::strerror(errno));
                    prefix_hash_valid = false;
                    return false;
                }
                completed += static_cast<size_t>(count);
            }
            sha256_update(&prefix_hash, buffer.data(), amount);
            offset += amount;
        }
        prefix_hash_valid = true;
        return true;
    }

    bool prefix_digest(std::array<uint8_t, 32> & digest, std::string & error) const {
        if (!prefix_hash_valid) {
            error = "durable prefix hash state is unavailable";
            return false;
        }
        sha256_t copy = prefix_hash;
        sha256_final(&copy, digest.data());
        return true;
    }

    bool fill_progress(StageState state, StageProgress & progress, std::string & error) const {
        progress = {};
        progress.state = state;
        progress.ticket_id = active_ticket;
        progress.residency_generation = active_generation;
        progress.verified_bytes = state == StageState::published ? active_spec.bytes : verified_bytes;
        progress.next_chunk = state == StageState::published
                ? static_cast<uint32_t>(active_spec.chunks.size()) : next_chunk;
        progress.manifest_sha256 = active_spec.manifest_sha256;
        if (state == StageState::published) {
            progress.prefix_sha256 = active_spec.sha256;
            return true;
        }
        return prefix_digest(progress.prefix_sha256, error);
    }

    bool quarantine_entry(
            const std::string & name,
            const std::array<uint8_t, 32> & digest,
            uint64_t ticket_id,
            std::string & error) {
        const std::string bad = quarantine_name(digest, ticket_id);
        if (unlinkat(directory.get(), bad.c_str(), 0) != 0 && errno != ENOENT) {
            error = "cannot replace quarantine entry: " + std::string(std::strerror(errno));
            return false;
        }
        if (renameat(directory.get(), name.c_str(), directory.get(), bad.c_str()) != 0) {
            error = "cannot quarantine store entry: " + std::string(std::strerror(errno));
            return false;
        }
        return sync_directory(error);
    }

    bool quarantine_active(std::string & error) {
        if (!active_file.valid()) {
            clear_active();
            return true;
        }
        const std::string name = part_name(active_spec.sha256);
        const auto digest = active_spec.sha256;
        const uint64_t ticket = active_ticket;
        active_file.reset();
        const bool ok = quarantine_entry(name, digest, ticket, error);
        clear_active();
        return ok;
    }

    enum class FinalState {
        absent,
        valid,
        invalid,
        error,
    };

    FinalState inspect_final(const StageObjectSpec & spec, Fd * result, std::string & error) const {
        const std::string name = final_name(spec.sha256);
        Fd file(openat(directory.get(), name.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
        if (!file.valid()) {
            if (errno == ENOENT) {
                return FinalState::absent;
            }
            error = "cannot open published object: " + std::string(std::strerror(errno));
            return FinalState::error;
        }
        struct stat st = {};
        if (!regular_owned_file(file.get(), st, error)) {
            return FinalState::error;
        }
        const uint64_t actual_bytes = static_cast<uint64_t>(st.st_size);
        if (spec.bytes == 0 || spec.bytes > limits.max_object_bytes || actual_bytes != spec.bytes) {
            error = "published object size differs from the requested bounded identity";
            return FinalState::error;
        }
        std::array<uint8_t, 32> digest = {};
        if (!sha256_fd_range(file.get(), 0, actual_bytes, digest, error)) {
            return FinalState::error;
        }
        if (digest != spec.sha256) {
            return FinalState::invalid;
        }
        if (result != nullptr) {
            *result = std::move(file);
        }
        return FinalState::valid;
    }

    bool used_bytes(uint64_t & bytes, std::string & error) const {
        bytes = 0;
        const int duplicate = openat(
                directory.get(), ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
        if (duplicate < 0) {
            error = "cannot duplicate store directory: " + std::string(std::strerror(errno));
            return false;
        }
        DIR * dir = fdopendir(duplicate);
        if (dir == nullptr) {
            close(duplicate);
            error = "cannot scan store directory: " + std::string(std::strerror(errno));
            return false;
        }
        bool ok = true;
        errno = 0;
        while (dirent * entry = readdir(dir)) {
            if (std::strcmp(entry->d_name, ".") == 0 || std::strcmp(entry->d_name, "..") == 0 ||
                std::strcmp(entry->d_name, ".lock") == 0) {
                continue;
            }
            if (!digest_name(entry->d_name, ".part") &&
                !digest_name(entry->d_name, ".gguf") &&
                !quarantine_file_name(entry->d_name)) {
                error = "unknown entry in private store directory";
                ok = false;
                break;
            }
            struct stat st = {};
            if (fstatat(directory.get(), entry->d_name, &st, AT_SYMLINK_NOFOLLOW) != 0) {
                error = "cannot inspect store entry: " + std::string(std::strerror(errno));
                ok = false;
                break;
            }
            if (!private_regular_stat(st) || st.st_nlink != 1) {
                error = "store entry is not an isolated private regular file";
                ok = false;
                break;
            }
            const uint64_t size = static_cast<uint64_t>(st.st_size);
            if (bytes > std::numeric_limits<uint64_t>::max() - size) {
                error = "store byte accounting overflow";
                ok = false;
                break;
            }
            bytes += size;
            errno = 0;
        }
        if (ok && errno != 0) {
            error = "store directory scan failed: " + std::string(std::strerror(errno));
            ok = false;
        }
        closedir(dir);
        return ok;
    }

    bool admit(
            const StageObjectSpec & spec,
            uint64_t existing_part,
            uint64_t allocated_part,
            std::string & error) const {
        uint64_t used = 0;
        if (!used_bytes(used, error) || used < existing_part) {
            if (error.empty()) error = "store byte accounting underflow";
            return false;
        }
        const uint64_t base = used - existing_part;
        if (base > limits.max_store_bytes || spec.bytes > limits.max_store_bytes - base) {
            error = "store quota would be exceeded";
            return false;
        }
        struct statvfs fs = {};
        if (fstatvfs(directory.get(), &fs) != 0) {
            error = "cannot query store free space: " + std::string(std::strerror(errno));
            return false;
        }
        const __uint128_t available_wide = static_cast<__uint128_t>(fs.f_bavail) * fs.f_frsize;
        const uint64_t available = available_wide > std::numeric_limits<uint64_t>::max()
                ? std::numeric_limits<uint64_t>::max() : static_cast<uint64_t>(available_wide);
        const uint64_t remaining = spec.bytes - std::min(allocated_part, spec.bytes);
        if (available < limits.min_free_bytes || remaining > available - limits.min_free_bytes) {
            error = "insufficient free space for staged object";
            return false;
        }
        return true;
    }
};

ArtifactStore::ArtifactStore() : impl_(new Impl) {}
ArtifactStore::~ArtifactStore() = default;

bool ArtifactStore::open(const std::string & root, const StoreLimits & limits, std::string & error) {
    if (enabled() || root.empty() || limits.max_store_bytes == 0 || limits.max_object_bytes == 0 ||
        limits.max_object_bytes > limits.max_store_bytes || limits.max_chunks == 0 ||
        limits.max_chunk_bytes == 0 || limits.max_chunk_bytes > k_max_stage_chunk_bytes) {
        error = "invalid store configuration";
        return false;
    }
    if (mkdir(root.c_str(), 0700) != 0 && errno != EEXIST) {
        error = "cannot create store directory: " + std::string(std::strerror(errno));
        return false;
    }
    Fd directory(::open(root.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
    if (!directory.valid()) {
        error = "cannot open store directory: " + std::string(std::strerror(errno));
        return false;
    }
    struct stat st = {};
    if (fstat(directory.get(), &st) != 0 || !S_ISDIR(st.st_mode) || st.st_uid != geteuid() ||
        (st.st_mode & 0077) != 0) {
        error = "store directory must be private and owned by the worker user";
        return false;
    }
    Fd lock(openat(directory.get(), ".lock", O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600));
    struct stat lock_st = {};
    if (!lock.valid() || !regular_owned_file(lock.get(), lock_st, error) ||
        flock(lock.get(), LOCK_EX | LOCK_NB) != 0) {
        if (error.empty()) {
            error = "store directory is already in use or cannot be locked";
        }
        return false;
    }
    impl_->limits = limits;
    impl_->root = root;
    impl_->directory = std::move(directory);
    impl_->lock = std::move(lock);
    uint64_t existing_bytes = 0;
    if (!impl_->reconcile_publish_pairs(error) ||
        !impl_->used_bytes(existing_bytes, error) ||
        existing_bytes > limits.max_store_bytes ||
        !impl_->sync_directory(error)) {
        if (error.empty()) error = "existing store exceeds the configured quota";
        impl_->directory.reset();
        impl_->lock.reset();
        return false;
    }
    return true;
}

bool ArtifactStore::begin(
        uint64_t ticket_id,
        uint64_t residency_generation,
        const StageObjectSpec & spec,
        StageProgress & progress,
        std::string & error) {
    if (!enabled() || ticket_id == 0 || residency_generation == 0 ||
        !validate_stage_spec(spec, impl_->limits, error)) {
        if (error.empty()) error = "invalid stage begin";
        return false;
    }

    if (impl_->active_file.valid() && !same_spec(impl_->active_spec, spec)) {
        error = "another staged object is active";
        return false;
    }

    if (!impl_->reconcile_publish_pair(spec.sha256, error)) {
        return false;
    }

    const uint64_t final_scan_start = now_us();
    Fd verified_final;
    const Impl::FinalState final = impl_->inspect_final(spec, &verified_final, error);
    if (final == Impl::FinalState::error) {
        return false;
    }
    if (final == Impl::FinalState::invalid) {
        if (!impl_->quarantine_entry(final_name(spec.sha256), spec.sha256, ticket_id, error)) {
            return false;
        }
        impl_->indeterminate_publications.erase(spec.sha256);
        impl_->verified_files.erase(spec.sha256);
    } else if (final == Impl::FinalState::valid) {
        const std::string stale_part = part_name(spec.sha256);
        if (unlinkat(impl_->directory.get(), stale_part.c_str(), 0) == 0) {
            if (!impl_->sync_directory(error)) {
                return false;
            }
        } else if (errno != ENOENT) {
            error = "cannot remove stale staging link: " + std::string(std::strerror(errno));
            return false;
        }
        if (!impl_->sync_directory(error)) {
            return false;
        }
        impl_->indeterminate_publications.erase(spec.sha256);
        if (!impl_->remember_verified(spec.sha256, spec.bytes, error)) {
            return false;
        }
        impl_->clear_active();
        impl_->active_spec = spec;
        impl_->active_ticket = ticket_id;
        impl_->active_generation = residency_generation;
        impl_->next_chunk = static_cast<uint32_t>(spec.chunks.size());
        impl_->verified_bytes = spec.bytes;
        impl_->active_file = std::move(verified_final);
        const bool ok = impl_->fill_progress(StageState::published, progress, error);
        progress.resume_scan_us = now_us() - final_scan_start;
        impl_->clear_active();
        return ok;
    }

    if (impl_->active_file.valid()) {
        if (!same_spec(impl_->active_spec, spec)) {
            error = "another staged object is active";
            return false;
        }
        impl_->active_ticket = ticket_id;
        impl_->active_generation = residency_generation;
        const uint64_t start = now_us();
        const bool ok = impl_->fill_progress(StageState::receiving, progress, error);
        progress.resume_scan_us = now_us() - start;
        return ok;
    }

    const std::string name = part_name(spec.sha256);
    bool created = false;
    int raw = openat(impl_->directory.get(), name.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW);
    if (raw < 0 && errno == ENOENT) {
        raw = openat(
                impl_->directory.get(), name.c_str(),
                O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
        created = raw >= 0;
    }
    Fd file(raw);
    if (!file.valid()) {
        error = "cannot open staging file: " + std::string(std::strerror(errno));
        return false;
    }
    struct stat st = {};
    if (!regular_owned_file(file.get(), st, error)) {
        return false;
    }
    const uint64_t existing = static_cast<uint64_t>(st.st_size);
    if (existing > spec.bytes) {
        error = "staging file exceeds the declared object size";
        return false;
    }
    const __uint128_t allocated_wide = static_cast<__uint128_t>(st.st_blocks) * 512;
    const uint64_t allocated = allocated_wide > spec.bytes
            ? spec.bytes : static_cast<uint64_t>(allocated_wide);
    if (!impl_->admit(spec, existing, allocated, error)) {
        return false;
    }
    if (created && !impl_->sync_directory(error)) {
        return false;
    }
    if (!reserve_fd(file.get(), spec.bytes, error) || !sync_fd(file.get(), true, error)) {
        return false;
    }

    const uint64_t scan_start = now_us();
    uint32_t next = 0;
    uint64_t verified = 0;
    while (next < spec.chunks.size()) {
        const StageChunkSpec & chunk = spec.chunks[next];
        if (chunk.offset + chunk.bytes > existing) {
            break;
        }
        std::array<uint8_t, 32> digest = {};
        if (!sha256_fd_range(file.get(), chunk.offset, chunk.bytes, digest, error)) {
            return false;
        }
        if (digest != chunk.sha256) {
            break;
        }
        verified = chunk.offset + chunk.bytes;
        ++next;
    }
    impl_->active_file = std::move(file);
    impl_->active_spec = spec;
    impl_->active_ticket = ticket_id;
    impl_->active_generation = residency_generation;
    impl_->next_chunk = next;
    impl_->verified_bytes = verified;
    impl_->metrics = {};
    if (!impl_->rebuild_prefix_hash(error)) {
        impl_->clear_active();
        return false;
    }
    if (!impl_->fill_progress(StageState::receiving, progress, error)) {
        impl_->clear_active();
        return false;
    }
    progress.resume_scan_us = now_us() - scan_start;
    return true;
}

bool ArtifactStore::put_chunk(
        uint64_t ticket_id,
        uint64_t residency_generation,
        uint32_t chunk_index,
        uint64_t offset,
        const uint8_t * data,
        size_t size,
        const std::array<uint8_t, 32> & chunk_sha256,
        StageProgress & progress,
        StageMetrics & metrics,
        std::string & error) {
    if (!enabled() || !impl_->active_file.valid() || ticket_id != impl_->active_ticket ||
        residency_generation != impl_->active_generation || chunk_index >= impl_->active_spec.chunks.size()) {
        error = "chunk does not name the active transfer";
        return false;
    }
    if (size > 0 && data == nullptr) {
        error = "chunk payload is null";
        return false;
    }
    const StageChunkSpec & chunk = impl_->active_spec.chunks[chunk_index];
    if (offset != chunk.offset || size != chunk.bytes || chunk_sha256 != chunk.sha256) {
        error = "chunk identity differs from the staged manifest";
        return false;
    }
    const uint64_t hash_start = now_us();
    const auto actual = sha256(data, size);
    impl_->metrics.chunk_hash_us += now_us() - hash_start;
    if (actual != chunk.sha256) {
        error = "chunk SHA-256 mismatch";
        return false;
    }

    if (chunk_index < impl_->next_chunk) {
        std::array<uint8_t, 32> stored = {};
        if (!sha256_fd_range(impl_->active_file.get(), chunk.offset, chunk.bytes, stored, error) ||
            stored != chunk.sha256) {
            error = "duplicate chunk does not match durable bytes";
            return false;
        }
        ++impl_->metrics.duplicate_chunks;
    } else {
        if (chunk_index != impl_->next_chunk || offset != impl_->verified_bytes) {
            error = "chunk is reordered or leaves a gap";
            return false;
        }
        const uint64_t write_start = now_us();
        if (!write_all_at(impl_->active_file.get(), data, size, offset, error)) {
            std::string rollback_error;
            if (!truncate_fd(impl_->active_file.get(), impl_->verified_bytes, rollback_error) ||
                !reserve_fd(impl_->active_file.get(), impl_->active_spec.bytes, rollback_error) ||
                !sync_fd(impl_->active_file.get(), true, rollback_error)) {
                impl_->quarantine_active(rollback_error);
                error += "; staging file quarantined after rollback failure";
            }
            return false;
        }
        impl_->metrics.write_us += now_us() - write_start;
        const uint64_t sync_start = now_us();
        if (!sync_fd(impl_->active_file.get(), true, error)) {
            std::string quarantine_error;
            impl_->quarantine_active(quarantine_error);
            error += "; staging file quarantined after durability failure";
            return false;
        }
        impl_->metrics.data_sync_us += now_us() - sync_start;
        if (!impl_->prefix_hash_valid) {
            error = "durable prefix hash state was lost";
            return false;
        }
        const uint64_t prefix_start = now_us();
        sha256_update(&impl_->prefix_hash, data, size);
        impl_->metrics.prefix_hash_us += now_us() - prefix_start;
        ++impl_->next_chunk;
        impl_->verified_bytes += size;
        ++impl_->metrics.accepted_chunks;
    }
    if (!impl_->fill_progress(StageState::receiving, progress, error)) {
        return false;
    }
    metrics = impl_->metrics;
    return true;
}

bool ArtifactStore::commit(
        uint64_t ticket_id,
        uint64_t residency_generation,
        const std::array<uint8_t, 32> & manifest_sha256,
        StageProgress & progress,
        StageMetrics & metrics,
        std::string & error) {
    if (!enabled() || !impl_->active_file.valid() || ticket_id != impl_->active_ticket ||
        residency_generation != impl_->active_generation ||
        manifest_sha256 != impl_->active_spec.manifest_sha256) {
        error = "commit does not name the active transfer";
        return false;
    }
    if (impl_->next_chunk != impl_->active_spec.chunks.size() ||
        impl_->verified_bytes != impl_->active_spec.bytes) {
        error = "cannot publish an incomplete staged object";
        return false;
    }
    struct stat st = {};
    if (!regular_owned_file(impl_->active_file.get(), st, error) ||
        static_cast<uint64_t>(st.st_size) != impl_->active_spec.bytes) {
        if (error.empty()) error = "staging file size changed before commit";
        return false;
    }
    const uint64_t verify_start = now_us();
    std::array<uint8_t, 32> digest = {};
    if (!sha256_fd_range(impl_->active_file.get(), 0, impl_->active_spec.bytes, digest, error)) {
        return false;
    }
    impl_->metrics.full_verify_us += now_us() - verify_start;
    if (digest != impl_->active_spec.sha256) {
        std::string quarantine_error;
        impl_->quarantine_active(quarantine_error);
        error = "full staged object SHA-256 mismatch";
        if (!quarantine_error.empty()) error += "; " + quarantine_error;
        return false;
    }
    const uint64_t file_sync_start = now_us();
    if (!sync_fd(impl_->active_file.get(), false, error)) {
        return false;
    }
    impl_->metrics.file_sync_us += now_us() - file_sync_start;

    const std::string part = part_name(impl_->active_spec.sha256);
    const std::string final = final_name(impl_->active_spec.sha256);
    const uint64_t publish_start = now_us();
    bool moved_part = rename_noreplace(
            impl_->directory.get(), part.c_str(), impl_->directory.get(), final.c_str(), error);
    if (!moved_part) {
        if (errno != EEXIST) return false;
        const Impl::FinalState existing = impl_->inspect_final(impl_->active_spec, nullptr, error);
        if (existing == Impl::FinalState::invalid) {
            if (!impl_->quarantine_entry(final, impl_->active_spec.sha256, impl_->active_ticket, error) ||
                !rename_noreplace(
                        impl_->directory.get(), part.c_str(),
                        impl_->directory.get(), final.c_str(), error)) {
                if (error.empty()) error = "cannot replace corrupt published object";
                return false;
            }
            moved_part = true;
        } else if (existing != Impl::FinalState::valid) {
            return false;
        }
    }
    impl_->indeterminate_publications.insert(impl_->active_spec.sha256);
    impl_->metrics.publish_us += now_us() - publish_start;
    if (moved_part) {
        // Persist the rename before sealing the inode. Otherwise a crash can
        // restore a read-only .part that cannot resume.
        if (!impl_->sync_directory(error, &impl_->metrics.directory_sync_us)) {
            return false;
        }
        const uint64_t seal_start = now_us();
        if (!make_read_only(impl_->active_file.get(), error) ||
            !sync_fd(impl_->active_file.get(), false, error)) {
            return false;
        }
        impl_->metrics.file_sync_us += now_us() - seal_start;
    }
    if (!impl_->sync_directory(error, &impl_->metrics.directory_sync_us)) {
        return false;
    }
    impl_->indeterminate_publications.erase(impl_->active_spec.sha256);
    if (!impl_->remember_verified(
                impl_->active_spec.sha256, impl_->active_spec.bytes, error)) {
        return false;
    }
    if (!moved_part) {
        if (unlinkat(impl_->directory.get(), part.c_str(), 0) != 0 && errno != ENOENT) {
            error = "cannot remove redundant staging object: " + std::string(std::strerror(errno));
            return false;
        }
        if (!impl_->sync_directory(error, &impl_->metrics.directory_sync_us)) {
            return false;
        }
    }

    progress = {};
    progress.state = StageState::published;
    progress.ticket_id = impl_->active_ticket;
    progress.residency_generation = impl_->active_generation;
    progress.verified_bytes = impl_->active_spec.bytes;
    progress.next_chunk = static_cast<uint32_t>(impl_->active_spec.chunks.size());
    progress.prefix_sha256 = impl_->active_spec.sha256;
    progress.manifest_sha256 = impl_->active_spec.manifest_sha256;
    metrics = impl_->metrics;
    impl_->clear_active();
    return true;
}

bool ArtifactStore::abort(
        uint64_t ticket_id,
        uint64_t residency_generation,
        bool quarantine,
        std::string & error) {
    if (!enabled() || !impl_->active_file.valid() || ticket_id != impl_->active_ticket ||
        residency_generation != impl_->active_generation) {
        error = "abort does not name the active transfer";
        return false;
    }
    if (quarantine && impl_->verified_bytes > 0) {
        return impl_->quarantine_active(error);
    }
    const std::string part = part_name(impl_->active_spec.sha256);
    impl_->active_file.reset();
    if (unlinkat(impl_->directory.get(), part.c_str(), 0) != 0 && errno != ENOENT) {
        error = "cannot remove staging file: " + std::string(std::strerror(errno));
        impl_->clear_active();
        return false;
    }
    const bool ok = impl_->sync_directory(error);
    impl_->clear_active();
    return ok;
}

bool ArtifactStore::lookup(
        uint64_t bytes,
        const std::array<uint8_t, 32> & digest,
        Fd & file,
        std::string & error) const {
    if (!enabled() || bytes == 0) {
        error = "dynamic store is not enabled or identity is empty";
        return false;
    }
    if (impl_->indeterminate_publications.count(digest) != 0) {
        error = "published object durability is indeterminate";
        return false;
    }
    error.clear();
    if (impl_->lookup_cached(bytes, digest, file, error)) {
        return true;
    }
    error.clear();
    StageObjectSpec spec;
    spec.bytes = bytes;
    spec.sha256 = digest;
    const Impl::FinalState state = impl_->inspect_final(spec, &file, error);
    if (state != Impl::FinalState::valid) {
        if (state == Impl::FinalState::absent) error = "published object is absent";
        if (state == Impl::FinalState::invalid) error = "published object failed verification";
        return false;
    }
    if (!impl_->remember_verified(digest, bytes, error)) {
        file.reset();
        return false;
    }
    return true;
}

bool ArtifactStore::active_progress(StageProgress & progress, std::string & error) const {
    if (!enabled() || !impl_->active_file.valid()) {
        return false;
    }
    return impl_->fill_progress(StageState::receiving, progress, error);
}

void ArtifactStore::detach_active() {
    if (impl_ != nullptr) {
        impl_->clear_active();
    }
}

bool ArtifactStore::enabled() const {
    return impl_ != nullptr && impl_->directory.valid();
}

} // namespace phone_pim
