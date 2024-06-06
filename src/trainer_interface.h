// Copyright 2016 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.!

#ifndef TRAINER_INTERFACE_H_
#define TRAINER_INTERFACE_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "leveldb/db.h"

#include "common.h"
#include "filesystem.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_processor.h"
#include "sentencepiece_trainer.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "util.h"

namespace sentencepiece
{

  template <typename K, typename V>
  std::vector<std::pair<K, V>> Sorted(const std::vector<std::pair<K, V>> &m)
  {
    std::vector<std::pair<K, V>> v = m;
    std::sort(v.begin(), v.end(),
              [](const std::pair<K, V> &p1, const std::pair<K, V> &p2)
              {
                return (p1.second > p2.second ||
                        (p1.second == p2.second && p1.first < p2.first));
              });
    return v;
  }

  template <typename K, typename V>
  std::vector<std::pair<K, V>> Sorted(const absl::flat_hash_map<K, V> &m)
  {
    std::vector<std::pair<K, V>> v(m.begin(), m.end());
    return Sorted(v);
  }

  class MultiFileSentenceIterator : public SentenceIterator
  {
  public:
    explicit MultiFileSentenceIterator(const std::vector<std::string> &files);
    ~MultiFileSentenceIterator() {}

    bool done() const override;
    void Next() override;
    const std::string &value() const override { return value_; }
    util::Status status() const override;

  private:
    void TryRead();

    bool read_done_ = false;
    size_t file_index_ = 0;
    std::vector<std::string> files_;
    std::string value_;
    std::unique_ptr<filesystem::ReadableFile> fp_;
  };

  // Base trainer class
  class TrainerInterface
  {
  public:
    using Sentence = std::pair<std::string, int64_t>;

    static const char32 kWSChar;
    static const char32 kUNKChar;
    static const char32 kUPPBoundaryChar;
    static const char kWSStr[];
    static const char kUNKStr[];
    static const char kUPPBoundaryStr[];

    TrainerInterface(const TrainerSpec &trainer_spec,
                     const NormalizerSpec &normalizer_spec,
                     const NormalizerSpec &denormalizer_spec)
        : trainer_spec_(trainer_spec),
          normalizer_spec_(normalizer_spec),
          denormalizer_spec_(denormalizer_spec),
          sentence_iterator_(nullptr),
          output_model_proto_(nullptr)
    {
      leveldb::Options options;
      options.create_if_missing = true;
      leveldb::Status status = leveldb::DB::Open(options, "sentences_db", &db_);
      if (!status.ok())
      {
        throw std::runtime_error("Failed to open LevelDB: " + status.ToString());
      }
    }

    virtual ~TrainerInterface()
    {
      delete db_;
    }

    // Loads sentence from `sentence_iterator` and stores the model
    // to `output_model_proto`.
    virtual util::Status Train(SentenceIterator *sentence_iterator,
                               ModelProto *output_model_proto)
    {
      sentence_iterator_ = sentence_iterator;
      output_model_proto_ = output_model_proto;
      return Train();
    }

    virtual util::Status Train() { return status(); }

    virtual util::Status status() const { return status_; }

    FRIEND_TEST(TrainerInterfaceTest, IsValidSentencePieceTest);
    FRIEND_TEST(TrainerInterfaceTest, OverrideSpecialPiecesTest);
    FRIEND_TEST(TrainerInterfaceTest, BytePiecesTest);
    FRIEND_TEST(TrainerInterfaceTest, SerializeTest);
    FRIEND_TEST(TrainerInterfaceTest, CharactersTest);

    // Loads all sentences from spec.input() or SentenceIterator.
    // It loads at most input_sentence_size sentences.
    util::Status LoadSentences();

    // Add a sentence to the database
    void addSentenceToDB(const Sentence &sentence)
    {
      std::string key = std::to_string(current_index_);
      std::string value = sentence.first + '\0' + std::to_string(sentence.second);

      leveldb::Status s = db_->Put(leveldb::WriteOptions(), key, value);
      if (!s.ok())
      {
        throw std::runtime_error("Failed to write to LevelDB: " + s.ToString());
      }

      ++current_index_;
    }

    // Get the Fi

    // Remove a sentence from the database
    void removeSentenceFromDB(size_t index)
    {
      std::string key = std::to_string(index);
      leveldb::Status s = db_->Delete(leveldb::WriteOptions(), key);
      if (!s.ok())
      {
        throw std::runtime_error("Failed to delete from LevelDB: " + s.ToString());
      }
    }

    // Load all sentences from the database
    void loadSentencesFromDB()
    {
      std::unique_ptr<leveldb::Iterator> it(db_->NewIterator(leveldb::ReadOptions()));
      for (it->SeekToFirst(); it->Valid(); it->Next())
      {
        std::string key = it->key().ToString();
        std::string value = it->value().ToString();
        size_t pos = value.find('\0');
        if (pos == std::string::npos)
        {
          throw std::runtime_error("Corrupted value in LevelDB");
        }
        std::string sentence = value.substr(0, pos);
        int64_t count = std::stoll(value.substr(pos + 1));
        std::cout << "Sentence: " << sentence << ", Count: " << count << std::endl;
      }

      if (!it->status().ok())
      {
        throw std::runtime_error("Failed to iterate LevelDB: " + it->status().ToString());
      }
    }

    // Retrieve a specific sentence from the database based on some criteria (e.g., index)
    Sentence getSentenceFromDB(size_t index) const
    {
      std::string key = std::to_string(index);
      std::string value;
      leveldb::Status s = db_->Get(leveldb::ReadOptions(), key, &value);
      if (!s.ok())
      {
        throw std::out_of_range("Index out of range or failed to read from LevelDB: " + s.ToString());
      }

      size_t pos = value.find('\0');
      if (pos == std::string::npos)
      {
        throw std::runtime_error("Corrupted value in LevelDB");
      }
      std::string sentence = value.substr(0, pos);
      int64_t count = std::stoll(value.substr(pos + 1));

      return {sentence, count};
    }

    // Update a specific sentence in the database based on some criteria (e.g., index)
    void updateSentenceInDB(size_t index, const Sentence &newSentence)
    {
      std::string key = std::to_string(index);
      std::string value = newSentence.first + '\0' + std::to_string(newSentence.second);

      leveldb::Status s = db_->Put(leveldb::WriteOptions(), key, value);
      if (!s.ok())
      {
        throw std::runtime_error("Failed to update LevelDB: " + s.ToString());
      }
    }

    size_t getDBSize() const
    {
      size_t count = 0;
      std::unique_ptr<leveldb::Iterator> it(db_->NewIterator(leveldb::ReadOptions()));
      for (it->SeekToFirst(); it->Valid(); it->Next())
      {
        ++count;
      }

      if (!it->status().ok())
      {
        throw std::runtime_error("Failed to iterate LevelDB: " + it->status().ToString());
      }

      return count;
    }

  protected:
    // Returns true if |piece| is valid sentence piece.
    // The result is affected by
    // max_sentencepiece_length, split_by_whitespace, split_by_unicode_script.
    bool IsValidSentencePiece(const string_util::UnicodeText &piece) const;

    // Splits all sentences by whitespaces and
    // replaces the tokenized string.
    // e.g.,
    //  [ ["hello world ", 1], ["hi world]" ] =>
    //  [ ["hello", 1], ["hi", 1], ["world", 2] ]
    void SplitSentencesByWhitespace();

    // Save model files into spec.model_prefix().
    util::Status Save() const;

    // Set of characters which must be included in the final vocab.
    // The value of this map stores the frequency.
    absl::flat_hash_map<char32, int64_t> required_chars_;

    // Final output pieces
    std::vector<std::pair<std::string, float>> final_pieces_;

    // Trainer spec.
    TrainerSpec trainer_spec_;

    // Normalizer spec
    NormalizerSpec normalizer_spec_;

    // Denormalizer spec
    NormalizerSpec denormalizer_spec_;

    // Reserved control pieces. e.g., <unk>, <s>, </s>.
    // key is vocab id.
    std::map<int, std::pair<std::string, ModelProto::SentencePiece::Type>> meta_pieces_;

    // Detect errors on initialization.
    util::Status status_;

    // Loads sentences from SentenceIterator if not null.
    SentenceIterator *sentence_iterator_ = nullptr;

    // Emits model to this proto instead of file.
    ModelProto *output_model_proto_ = nullptr;
    leveldb::DB *db_;

  private:
    // Serialize final_pieces_ to |model_proto|.
    util::Status Serialize(ModelProto *model_proto) const;

    // Saves the best sentence split with the current model for debugging.
    util::Status SaveSplits(absl::string_view filename) const;

    // Saves model file.
    util::Status SaveModel(absl::string_view filename) const;

    // Saves vocabulary file for NMT.
    util::Status SaveVocab(absl::string_view filename) const;

    // Initializes `meta_pieces_` from TrainerSpec.
    util::Status InitMetaPieces();

    // Randomly sampled raw sentences for self-testing.
    std::vector<std::string> self_test_samples_;

    // LevelDB database instance
    size_t current_index_ = 0; // to keep track of the last index used in LevelDB
  };
} // namespace sentencepiece
#endif // TRAINER_INTERFACE_H_
