/*
 * CSV_Reader.hpp
 *
 *  Created on: 16.01.2019
 *      Author: tomlucas
 */

#include "CSV_Reader.hpp"
namespace zavi
::csv_reader {

	bool readLineToPointer(std::ifstream &csv, double * result, const unsigned int size,const char delim) {
		std::string line;
		if (csv.good()) {
			getline(csv, line);
			auto string_list=splitString(line,delim);
			if(string_list.size() < size) {
				std::cout<< "Line of file did not contain enough data. Expected: "<<size << " Got: " <<string_list.size() << std::endl;
				return false;
			}
			for (size_t i=0; i < string_list.size(); i++) {
				result[i]=atof(string_list[i].c_str());
			}
			return true;
		}
		else {
			std::cout<< "CSV File is empty"<<std::endl;
			return false;
		}
	}

	std::vector<std::string> splitString(const std::string & to_split, const char delimiter) {
		std::stringstream split_stream(to_split);
		std::string segment;
		std::vector<std::string> seglist;
		while (std::getline(split_stream, segment, delimiter)) {
			if (!segment.empty())
			seglist.push_back(segment);
		}
		return seglist;

	}

	std::vector<double> parseFloatVector(const std::vector<std::string> & vector) {
		std::vector<double> floats;
		for (const std::string & number : vector) {
			floats.push_back(atof(number.c_str()));
		}
		return floats;
	}
	std::vector<int> parseIntVector(const std::vector<std::string> & vector){
	std::vector<int> ints;
		for (const std::string & number : vector) {
			ints.push_back(atoi(number.c_str()));
		}
		return ints;
	}

}
//zavi::csv_reader

