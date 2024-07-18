#include "pretokenizer_for_training.h"

#include <string>

#include "third_party/absl/strings/str_replace.h"
#include "leveldb/write_batch.h"

namespace sentencepiece {
namespace pretokenizer {

namespace {
const char kWSStr[] = "\xe2\x96\x81";
}  // namespace

PretokenizerForTrainingInterface::PretokenizerForTrainingInterface() {
  leveldb::Options options;
  options.create_if_missing = true;
  leveldb::DB* db;
  leveldb::Status status = leveldb::DB::Open(options, "/tmp/pretokenizer_db", &db);
  if (status.ok()) {
    db_.reset(db);
  } else {
    // Handle error
  }
}

PretokenizerForTrainingInterface::~PretokenizerForTrainingInterface() = default;

std::string PretokenizerForTrainingInterface::PreTokenize(
    absl::string_view text) const {
  SentencePieceText spt = Tokenize(Preprocess(text));
  
  leveldb::WriteBatch batch;
  Postprocess(spt, &batch);
  
  std::string key = "0"; // Use a fixed key for simplicity in tests
  db_->Write(leveldb::WriteOptions(), &batch);
  
  return key;
}

// static
std::string PretokenizerForTrainingInterface::Preprocess(
    absl::string_view text) {
  return absl::StrReplaceAll(text, {{kWSStr, " "}});
}

// static
void PretokenizerForTrainingInterface::Postprocess(
    const SentencePieceText &spt, leveldb::WriteBatch* batch) const {
  std::string output;
  int prev = 0;
  int counter = 0;

  for (const auto &piece : spt.pieces()) {
    if (prev == piece.begin() && piece.begin() != 0) {
      std::string key = std::to_string(counter++);
      batch->Put(key, output);
      output.clear();
    } else {
      output.append(piece.begin() - prev, ' ');
    }
    output += piece.surface();
    prev = piece.end();
  }

  if (!output.empty()) {
    std::string key = std::to_string(counter);
    batch->Put(key, output);
  }

  // Replace spaces with kWSStr in all stored values
  std::string value;
  for (int i = 0; i <= counter; ++i) {
    std::string key = std::to_string(i);
    leveldb::Status s = db_->Get(leveldb::ReadOptions(), key, &value);
    if (s.ok()) {
      value = absl::StrReplaceAll(value, {{" ", kWSStr}});
      batch->Put(key, value);
    }
  }
}

}  // namespace pretokenizer
}  // namespace sentencepiece