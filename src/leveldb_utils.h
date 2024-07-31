#ifndef LEVELDB_UTILS_H_
#define LEVELDB_UTILS_H_

#include <leveldb/db.h>
#include "util.h"  // For util::Status

namespace sentencepiece {

class LevelDBManager {
 public:
  LevelDBManager();
  ~LevelDBManager();

  util::Status OpenDB(const std::string& db_path);
  util::Status CloseDB();
  
  leveldb::DB* GetDB() { return db_; }

 private:
  leveldb::DB* db_;
};

// Global instance
extern LevelDBManager g_leveldb_manager;

}  // namespace sentencepiece

#endif  // LEVELDB_UTILS_H_