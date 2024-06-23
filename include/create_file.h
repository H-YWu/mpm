#ifndef CHAINS_CREATE_FILE_H_
#define CHAINS_CREATE_FILE_H_

#include <string>

namespace chains {

std::string createDirectoryWithPrefixAndCurrentTime(const std::string& prefix);

std::string createFileWithPaddingNumberNameInDirectory(int number, size_t padding, std::string directory);

}

#endif  // CHAINS_CREATE_FILE_H_