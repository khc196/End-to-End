#include "Lane_Detector.hpp"
#include <stdio.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string.h>
#include <string>  

#include <sys/shm.h>
#include <sys/sem.h>
#include <sys/ipc.h>
#include <unistd.h>

using namespace cv;
using namespace std;

int main(int argc, char* argv) 
{ 
    int shmid;
    key_t skey;
    void *shared_memory;
    skey = 5678;
    
    shmid = shmget((key_t) skey, sizeof(unsigned char) * 640 * 480 * 3, 0777);
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
        Mat origin_img(640, 480, CV_8UC3, img);
        ld->operate(origin_img);
    }
    delete ld;
}

