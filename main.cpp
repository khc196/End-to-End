#include "Lane_Detector.hpp"
#include <stdio.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string.h>
#include <string>  

using namespace cv;
using namespace std;

int main(int argc, char* argv) 
{ 
    Lane_Detector* ld = new Lane_Detector();
    ld->init();
    DIR            *dir_info;
    struct dirent  *dir_entry;
    const char* origin_data_dir = "captures/captures_e";
    const char* result_data_path = "generated_data"; 
    char origin_data_path[64];  
    snprintf(origin_data_path, 64, "%s/%s", getenv("HOME"),origin_data_dir);
    printf("original data path : %s\n", origin_data_path);
    dir_info = opendir(origin_data_path);
    if ( NULL != dir_info)
    {
        mkdir(result_data_path, 0755);
        while( dir_entry   = readdir( dir_info))  
        {
            if(!strcmp(dir_entry->d_name, ".") || !strcmp(dir_entry->d_name, "..")) 
            {
                continue;
            }
            DIR            *l_dir_info;
            struct dirent  *l_dir_entry;
            char buffer_1[64];
            char buffer_2[64];
            snprintf(buffer_1, 64, "%s/%s", result_data_path, dir_entry->d_name);
            printf("%s\n", dir_entry->d_name);
            mkdir(buffer_1, 0755);
            snprintf(buffer_2, 64, "%s/%s", origin_data_path, dir_entry->d_name); 
            l_dir_info = opendir(buffer_2);
            if( NULL != l_dir_info) 
            {
                while(l_dir_entry = readdir(l_dir_info))
                {
                    if(!strcmp(l_dir_entry->d_name, ".") || !strcmp(l_dir_entry->d_name, "..")) 
                    {
                        continue;
                    }
                    int frame; 
                    char buffer_3[64];
                    snprintf(buffer_3, 64, "%s/%s", buffer_2, l_dir_entry->d_name);
                    printf("current image : %s\n", buffer_3);
                    Mat origin_img = imread(buffer_3, CV_LOAD_IMAGE_COLOR);
                    frame = atoi(strtok(l_dir_entry->d_name, "#"));
                    printf("frame : %d\n", frame);
                    ld->operate(origin_img);
                    //char output_file_name[128];
                    //snprintf(output_file_name, 128, "%s/%05d#%d#%d#%d#%d#%d#%d#%d#%d.png", buffer_1, frame, ld->p1.x, ld->p1.y, ld->p2.x, ld->p2.y, ld->p3.x, ld->p3.y, ld->p4.x, ld->p4.y);
                    //printf("output file name : %s\n", output_file_name);
                    //imwrite(output_file_name, origin_img, vector<int>(IMWRITE_PNG_COMPRESSION, 3));
                }
            }
        }
        closedir( dir_info);
    }   
    delete ld;
}

