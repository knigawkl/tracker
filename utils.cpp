#include <sstream>
#include <iostream>

#include "utils.hpp"

void make_tmp_dirs(std::string tmp_folder) 
{
    std::stringstream ss;
    ss << "mkdir " << tmp_folder << "/img " << tmp_folder << "/csv";
    std::string mkdir_command = ss.str();
    std::cout << "Executing: " << mkdir_command << std::endl;
    system(mkdir_command.c_str());
}

void clear_tmp(std::string tmp_folder) 
{
    std::stringstream ss;
    ss << "exec rm -r " << tmp_folder << "/*";
    std::string del_command = ss.str();
    std::cout << "Executing: " << del_command << std::endl;
    system(del_command.c_str());
}

void mv(std::string what, std::string where)
{
    std::stringstream ss;
    ss << "mv " << what << " " << where;
    std::string mv_command = ss.str();
    std::cout << "Executing: " << mv_command << std::endl;
    system(mv_command.c_str());
}

void cp(std::string what, std::string where)
{
    std::stringstream ss;
    ss << "cp " << what << " " << where;
    std::string cp_command = ss.str();
    std::cout << "Executing: " << cp_command << std::endl;
    system(cp_command.c_str());
}
