#include "leveldb_utils.h"

namespace sentencepiece {

LevelDBManager g_leveldb_manager;

LevelDBManager::LevelDBManager() : db_(nullptr) {}

LevelDBManager::~LevelDBManager() {
  CloseDB();
}

util::Status LevelDBManager::OpenDB(const std::string& db_path) {
  if (db_ != nullptr) {
    return util::OkStatus();  // Already open
  }
  
  leveldb::Options options;
  options.create_if_missing = true;
  leveldb::Status status = leveldb::DB::Open(options, db_path, &db_);
  if (!status.ok()) {
    return util::InternalError("Failed to open LevelDB: " + status.ToString());
  }
  return util::OkStatus();
}

util::Status LevelDBManager::CloseDB() {
  if (db_ != nullptr) {
    delete db_;
    db_ = nullptr;
  }
  return util::OkStatus();
}

}  // namespace sentencepiece