#pragma once

#include "phone_pim_protocol.h"
#include "phone_pim_socket.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace phone_pim {

struct StageChunkSpec {
    uint32_t index = 0;
    uint64_t offset = 0;
    uint32_t bytes = 0;
    std::array<uint8_t, 32> sha256 = {};
};

struct StageObjectSpec {
    uint64_t bytes = 0;
    std::array<uint8_t, 32> sha256 = {};
    uint32_t chunk_bytes = 0;
    std::vector<StageChunkSpec> chunks;
    std::array<uint8_t, 32> manifest_sha256 = {};
};

enum class StageState : uint32_t {
    receiving = 1,
    published = 2,
};

struct StageProgress {
    StageState state = StageState::receiving;
    uint64_t ticket_id = 0;
    uint64_t residency_generation = 0;
    uint64_t verified_bytes = 0;
    uint32_t next_chunk = 0;
    std::array<uint8_t, 32> prefix_sha256 = {};
    std::array<uint8_t, 32> manifest_sha256 = {};
    uint64_t resume_scan_us = 0;
};

struct StageMetrics {
    uint64_t chunk_hash_us = 0;      // data-only manifest verify of each accepted chunk
    uint64_t write_us = 0;
    uint64_t data_sync_us = 0;
    uint64_t full_verify_us = 0;
    uint64_t file_sync_us = 0;
    uint64_t publish_us = 0;
    uint64_t directory_sync_us = 0;
    uint64_t accepted_chunks = 0;
    uint64_t duplicate_chunks = 0;
    // Measurement-only, NOT serialized over the wire (stage_commit_payload writes only the
    // nine fields above). Data-only rolling durable-prefix hash advance time.
    uint64_t prefix_hash_us = 0;
};

struct StoreLimits {
    uint64_t max_store_bytes = 16ULL * 1024 * 1024 * 1024;
    uint64_t max_object_bytes = 16ULL * 1024 * 1024 * 1024;
    uint64_t min_free_bytes = 256ULL * 1024 * 1024;
    uint32_t max_chunks = k_max_stage_chunks;
    uint64_t max_chunk_bytes = k_max_stage_chunk_bytes;
};

std::array<uint8_t, 32> stage_manifest_sha256(const StageObjectSpec & spec);
bool validate_stage_spec(const StageObjectSpec & spec, const StoreLimits & limits, std::string & error);

class ArtifactStore {
public:
    ArtifactStore();
    ~ArtifactStore();

    ArtifactStore(const ArtifactStore &) = delete;
    ArtifactStore & operator=(const ArtifactStore &) = delete;

    bool open(const std::string & root, const StoreLimits & limits, std::string & error);
    bool begin(
            uint64_t ticket_id,
            uint64_t residency_generation,
            const StageObjectSpec & spec,
            StageProgress & progress,
            std::string & error);
    bool put_chunk(
            uint64_t ticket_id,
            uint64_t residency_generation,
            uint32_t chunk_index,
            uint64_t offset,
            const uint8_t * data,
            size_t size,
            const std::array<uint8_t, 32> & chunk_sha256,
            StageProgress & progress,
            StageMetrics & metrics,
            std::string & error);
    bool commit(
            uint64_t ticket_id,
            uint64_t residency_generation,
            const std::array<uint8_t, 32> & manifest_sha256,
            StageProgress & progress,
            StageMetrics & metrics,
            std::string & error);
    bool abort(
            uint64_t ticket_id,
            uint64_t residency_generation,
            bool quarantine,
            std::string & error);
    bool lookup(
            uint64_t bytes,
            const std::array<uint8_t, 32> & digest,
            Fd & file,
            std::string & error) const;

    bool active_progress(StageProgress & progress, std::string & error) const;
    void detach_active();
    bool enabled() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace phone_pim
