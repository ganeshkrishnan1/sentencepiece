#include <leveldb/db.h>
#include <iostream>

int main() {
    leveldb::DB* db;
    leveldb::Options options;
    options.create_if_missing = false;

    // Open the LevelDB database
    leveldb::Status status = leveldb::DB::Open(options, "/path/to/leveldb-directory", &db);
    if (!status.ok()) std::cerr << status.ToString() << std::endl;

    // Iterate through key-value pairs
    leveldb::Iterator* it = db->NewIterator(leveldb::ReadOptions());
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
        std::cout << it->key().ToString() << ": " << it->value().ToString() << std::endl;
    }
    delete it;
    delete db;
    return 0;
}