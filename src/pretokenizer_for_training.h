#ifndef PRETOKENIZER_FOR_TRAINING_H_
#define PRETOKENIZER_FOR_TRAINING_H_

#include <memory>
#include <string>

#include "common.h"
#include "sentencepiece.pb.h"
#include "sentencepiece_processor.h"
#include "third_party/absl/strings/string_view.h"
#include "leveldb/db.h"

namespace sentencepiece {
namespace pretokenizer {

class PretokenizerForTrainingInterface {
 public:
  PretokenizerForTrainingInterface();
  virtual ~PretokenizerForTrainingInterface();
  virtual util::Status status() const = 0;

  leveldb::DB* GetDB() const {
    return db_.get();
  }

  // Puts kUPPBoundaryStr before and after the pre-tokenizer's segmentation
  // when there are no spaces between these tokens.
  // Returns a unique key to access the pretokenized result in LevelDB.
  std::string PreTokenize(absl::string_view text) const;

  // Returns pre-tokenized result.
  // Note that the pre-tokenized constraint is specified with the
  // byte offsets (SentencePiece::begin, SentencePiece::end) over
  // the input text.
  virtual SentencePieceText Tokenize(absl::string_view text) const = 0;

 protected:
  std::unique_ptr<leveldb::DB> db_;

 private:
  static std::string Preprocess(absl::string_view text);
  void Postprocess(const SentencePieceText &spt, leveldb::WriteBatch* batch) const;
};

}  // namespace pretokenizer
}  // namespace sentencepiece

#endif  // PRETOKENIZER_FOR_TRAINING_H_