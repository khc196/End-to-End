#include "Lane_Detector.hpp"
#include <stdio.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string.h>
#include <string>  
#include <math.h>


#include <sys/shm.h>
#include <sys/sem.h>
#include <sys/ipc.h>
#include <unistd.h>

using namespace cv;
using namespace std;

int main(int argc, char* argv) 
{ 
    int shmid;
    const int skey = 5678;
    void *shared_memory;
    int size_of_shared_memory = sizeof(unsigned char*) * 640 * 480 * 3;
    shmid = shmget((key_t) skey, size_of_shared_memory + sizeof(int) * 2, 0777);
    if (shmid == -1) {
		perror("shmget failed :");
        exit(1);
	}
    shared_memory = shmat(shmid, (void *) 0, 0);
	if (!shared_memory) {
		perror("shmat failed");
		exit(1);
	}

    Lane_Detector* ld = new Lane_Detector();
    ld->init();
    while(true){
        unsigned char* img = (unsigned char*)shared_memory;
        int* torcs_steer = (int*)(shared_memory + size_of_shared_memory);
        int* torcs_lock = (int*)(shared_memory + size_of_shared_memory + sizeof(int));
        if(*torcs_lock == 1) {
            Mat origin_img(480, 640, CV_8UC3, img);
            Mat flipped_img;
            memcpy(origin_img.data, img, sizeof(unsigned char) * 640 * 480 * 3);
            flip(origin_img, flipped_img, 0);
            ld->operate(flipped_img);
            vector<double> left_lane = ld->get_left_lane();
            vector<double> right_lane = ld->get_right_lane();
            int steer;
            int position = (left_lane[0] + right_lane[0]) / 2; 
            int error = 70 - position;
            steer = tan((left_lane[1] + right_lane[1])) * 20 + error*2.5;
            printf("tan : %f\n", tan((left_lane[1] + right_lane[1])));
            if(steer > 100) {
                steer = 100;
            }
            else if(steer < -100){
                steer = -100;
            }
            printf("position : %d\n", error);
            printf("steer : %d\n", steer);
            *torcs_steer = steer;
            *torcs_lock = 0;
        }
    }
    delete ld;
}

