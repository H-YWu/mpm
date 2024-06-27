#ifndef CHAINS_CREATE_FILE_H_
#define CHAINS_CREATE_FILE_H_

#include <string>

namespace chains {

std::string createNewDirectoryInParentDirectoryWithPrefixAndCurrentTime(const std::string &parentDirectoryPath, const std::string& prefix);

std::string createFileWithPaddingNumberNameInDirectory(int number, size_t padding, const std::string &directory, const std::string &fileExtension);

}

#endif  // CHAINS_CREATE_FILE_H_