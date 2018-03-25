#ifndef TEST_HELPER_H
#define TEST_HELPER_H

#include <iostream>
#include <sstream>

template<class T>
std::string check(const T& result, const T& expectation) {
  std::stringstream checkString;
  checkString << "Result is: " << result << ", expectation is: " << expectation;
  if(result == expectation) {
    checkString << "\tCORRECT";
  } else {
    checkString << "\tERROR";
  }
  return checkString.str();
}

#endif // TEST_HELPER_H
