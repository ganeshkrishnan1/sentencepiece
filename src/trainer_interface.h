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

// /*
// spm_train-- input = sample.txt-- model_prefix = wm-- vocab_size = 32000 --character_coverage = 1.0 --model_type = unigram-- max_sentence_length = 4096 --train_extremely_large_corpus = true
// */

#ifndef TRAINER_INTERFACE_H_
#define TRAINER_INTERFACE_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "common.h"
#include "filesystem.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_processor.h"
#include "sentencepiece_trainer.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "util.h"
#include "leveldb/db.h"
#include "leveldb/write_batch.h"
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <unicode/uscript.h> // Assuming you have this header for script handling
#include "trainer_spec.h"   // Assuming you have this header for TrainerSpec
#include "normalizer_spec.h"

namespace sentencepiece
{

  template <typename K, typename V>
  std::vector<std::pair<K, V>> Sorted(const std::vector<std::pair<K V>> &m)
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
    using Sentences = leveldb::DB *;

    static const char32_t kWSChar;
    static const char32_t kUNKChar;
    static const char32_t kUPPBoundaryChar;
    static const char kWSStr[];
    static const char kUNKStr[];
    static const char kUPPBoundaryStr[];

    TrainerInterface(const TrainerSpec &trainer_spec,
                     const NormalizerSpec &normalizer_spec,
                     const NormalizerSpec &denormalizer_spec)
        : trainer_spec_(trainer_spec),
          normalizer_spec_(normalizer_spec),
          denormalizer_spec_(denormalizer_spec)
    {
      status_ = VerifySpec(trainer_spec_);
      if (status_.ok())
        status_ = InitMetaPieces();

      leveldb::Options options;
      options.create_if_missing = true;
      leveldb::Status status = leveldb::DB::Open(options, "sentences_db", &sentences_db_);
      if (!status.ok())
      {
        throw std::runtime_error("Failed to open LevelDB: " + status.ToString());
      }
    }

    virtual ~TrainerInterface()
    {
      delete sentences_db_;
    }

    bool IsValidSentencePiece(const string_util::UnicodeText &sentencepiece) const
    {
      // Returns false if the length of piece is invalid.
      if (sentencepiece.empty() ||
          sentencepiece.size() > static_cast<size_t>(trainer_spec_.max_sentencepiece_length()))
      {
        return false;
      }

      constexpr unicode_script::ScriptType kAnyType = static_cast<unicode_script::ScriptType>(-1);
      unicode_script::ScriptType prev_script = kAnyType;
      bool all_whitespace_piece = std::all_of(sentencepiece.begin(), sentencepiece.end(),
                                              [](char32_t c)
                                              { return c == kWSChar; });

      for (size_t pos = 0; pos < sentencepiece.size(); ++pos)
      {
        const char32_t c = sentencepiece[pos];
        if (c == kUNKChar)
        { // UNK must not be included
          return false;
        }
        if (c == 0x0000)
        { // NULL is not allowed for Darts (TRIE).
          return false;
        }
        if (c == kUPPBoundaryChar)
        {
          return false;
        }
        if (c == 0x0020)
        {
          std::cerr << "space must not be included in normalized string." << std::endl;
          return false;
        }
        if (!string_util::IsValidCodepoint(c))
        {
          return false;
        }

        if (c == kWSChar)
        {
          // Only allows whitespace to appear as a prefix of piece unless
          // allow_whitespace_only_pieces is True.
          // When split_by_whitespace is false, we allow whitespaces to
          // appear in the middle, "foo_bar", but do not allow them
          // to appear as suffix, "foo_bar_".
          // Regardless of the setting of split_by_whitespace,
          // whitespace is treated as a prefix/infix of symbol or
          // independent symbol, unless allow_whitespace_only_pieces() is true,
          // in which case whitespace only pieces can occur.
          if (!trainer_spec_.allow_whitespace_only_pieces() || !all_whitespace_piece)
          {
            if (trainer_spec_.treat_whitespace_as_suffix())
            {
              if ((trainer_spec_.split_by_whitespace() && pos < sentencepiece.size() - 1) ||
                  (!trainer_spec_.split_by_whitespace() && pos < sentencepiece.size() - 1 && pos == 0))
              {
                return false;
              }
            }
            else
            {
              if ((trainer_spec_.split_by_whitespace() && pos > 0) ||
                  (!trainer_spec_.split_by_whitespace() && pos > 0 && pos == sentencepiece.size() - 1))
              {
                return false;
              }
            }
          }
        }
        else
        {
          auto s = unicode_script::GetScript(c);

          // Merge Hiragana/Katakana into Han.
          if (s == unicode_script::U_Hiragana || s == unicode_script::U_Katakana || c == 0x30FC)
          { // long vowel sound (Katakana) should be Katakana
            s = unicode_script::U_Han;
          }
          else if (s == unicode_script::U_Inherited)
          {
            s = prev_script;
          }

          if (!trainer_spec_.split_by_number() && is_unicode_decimal_number(c))
          {
            s = kAnyType;
          }

          if (trainer_spec_.split_digits() && is_unicode_decimal_number(c))
          {
            if (sentencepiece.size() > 1)
              return false;
          }

          // Do not allow a piece to include multiple Unicode scripts
          // when split_by_unicode_script() is true (default = true).
          if (trainer_spec_.split_by_unicode_script() && s != kAnyType &&
              prev_script != kAnyType && prev_script != s)
          {
            return false;
          }

          prev_script = s;
        }
      }
      return true;
    }

    // Serializes a Sentence into a string
    std::string serializeSentence(const Sentence &sentence) const
    {
      std::ostringstream oss;
      size_t strSize = sentence.first.size();
      oss.write(reinterpret_cast<const char *>(&strSize), sizeof(strSize));
      oss.write(sentence.first.data(), strSize);
      oss.write(reinterpret_cast<const char *>(&sentence.second), sizeof(sentence.second));
      return oss.str();
    }

    // Deserializes a string into a Sentence
    Sentence deserializeSentence(const std::string &data) const
    {
      std::istringstream iss(data);
      size_t strSize;
      iss.read(reinterpret_cast<char *>(&strSize), sizeof(strSize));
      std::string str(strSize, '\0');
      iss.read(&str[0], strSize);
      int64_t number;
      iss.read(reinterpret_cast<char *>(&number), sizeof(number));
      return {str, number};
    }

    // Add a sentence to the LevelDB
    void addSentenceToDB(const Sentence &sentence, const std::string &key)
    {
      std::string serializedSentence = serializeSentence(sentence);
      leveldb::Status status = sentences_db_->Put(leveldb::WriteOptions(), key, serializedSentence);
      if (!status.ok())
      {
        throw std::runtime_error("Failed to write to LevelDB: " + status.ToString());
      }
    }

    // Retrieve a sentence from the LevelDB
    Sentence getSentenceFromDB(const std::string &key)
    {
      std::string value;
      leveldb::Status status = sentences_db_->Get(leveldb::ReadOptions(), key, &value);
      if (!status.ok())
      {
        throw std::runtime_error("Failed to read from LevelDB: " + status.ToString());
      }
      return deserializeSentence(value);
    }

    // Update a sentence in the LevelDB
    void updateSentenceInDB(const std::string &key, const Sentence &newSentence)
    {
      addSentenceToDB(newSentence, key); // Since LevelDB Put will overwrite existing entry
    }

    // Delete a sentence from the LevelDB
    void deleteSentenceFromDB(const std::string &key)
    {
      leveldb::Status status = sentences_db_->Delete(leveldb::WriteOptions(), key);
      if (!status.ok())
      {
        throw std::runtime_error("Failed to delete from LevelDB: " + status.ToString());
      }
    }

  protected:
    // Other existing methods...

  private:
    leveldb::DB *sentences_db_; // LevelDB database for storing sentences
    TrainerSpec trainer_spec_;
    NormalizerSpec normalizer_spec_;
    NormalizerSpec denormalizer_spec_;
    util::Status status_;
  };

} // namespace sentencepiece
#endif // TRAINER_INTERFACE_H_
