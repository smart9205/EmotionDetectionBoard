#include <string.h>
#include <stdlib.h>
#include <imp/imp_log.h>
#include <imp/imp_common.h>
#include <imp/imp_system.h>
#include <imp/imp_framesource.h>
#include <imp/imp_ivs.h>
#include <imp/imp_ivs_move.h>
#include "sample-common.h"
#include "inference_nv12.h"

#define TAG "Sample-IVS-unbind-move"
#include <signal.h> 
#include <pthread.h> 
#include <vector>
#include <dirent.h>
#include <ctime>
#include <sstream>
#include <fstream>
#include <cstdlib> 
#include <iostream>  

#define FACE_MODEL_PATH "face.bin"
#define EMO_MODEL_PATH "emo.bin"

extern struct chn_conf chn[];
std::unique_ptr<venus::BaseNet> face_net;
std::unique_ptr<venus::BaseNet> emo_net;

pthread_mutex_t rec_mutex = PTHREAD_MUTEX_INITIALIZER;  
volatile bool isRec = false;  
volatile bool isMerge = false;
pthread_t rec_thread, merge_thread;

static int sample_venus_init()
{
	int ret = 0;
 
    ret = venus::venus_init();
    if (0 != ret) {
        fprintf(stderr, "venus init failed.\n");
		return -1;
    }

	face_net = venus::net_create(TensorFormat::NV12);
	// emo_net = venus::net_create(TensorFormat::NV12);
	emo_net = venus::net_create(TensorFormat::NHWC);

    ret = face_net->load_model(FACE_MODEL_PATH);
    ret = emo_net->load_model(EMO_MODEL_PATH);

	return 0;
}
static int sample_venus_deinit()
{
    int ret = venus::venus_deinit();
    if (0 != ret) {
        fprintf(stderr, "venus deinit failed.\n");
    }

	return ret;
}

// Global variable to control the loop  
volatile sig_atomic_t keep_running = 1;  

// Signal handler function to handle SIGINT  
void handle_sigint(int sig) {  
    keep_running = 0;  
}

bool createTempFileList(const std::vector<std::string>& videoFiles, const std::string& listFilePath) {  
    std::ofstream listFile(listFilePath);  
    if (!listFile.is_open()) {  
        std::cerr << "Error: Could not open file list for writing." << std::endl;  
        return false;  
    }  
    for (const auto& filePath : videoFiles) {  
        listFile << "file '" << filePath << "'\n";  
    }  
    listFile.close();  
    return true;  
} 

// Function to handle video recording  
void* recording_thread_func(void* arg) {  
    int n_clips = *((int*)arg);  
    printf("Recording started for clip %d\n", n_clips);  
    sample_get_video_stream(n_clips, 0);  

	// Wait until the main loop signals to stop recording  
    while (1) {  
        pthread_mutex_lock(&rec_mutex);  
        if (!isRec) { // Check if the main loop has signaled to stop  
            printf("Stopping recording for clip %d\n", n_clips);  
            sample_get_video_stream(n_clips, 1);  
            pthread_mutex_unlock(&rec_mutex);  
            break; // Exit the loop and finish the thread  
        }  
        pthread_mutex_unlock(&rec_mutex);  
        usleep(100000); // Sleep for a short duration to avoid busy waiting  
    }
  
    // pthread_mutex_lock(&rec_mutex);  
    // if (isRec) {  
    //     printf("Stopping recording for clip %d\n", n_clips);  
    //     sample_get_video_stream(n_clips, 1);  
    //     isRec = false;  
    // }  
    // pthread_mutex_unlock(&rec_mutex);  
  
    return NULL;  
} 

void* merge_recording_thread_func(void* arg) {  
    int nClips = *((int*)arg);  
    printf("Merging recording files......\n");  
  
    std::vector<std::string> videoFiles;  
    const char* directoryPath = STREAM_FILE_PATH_PREFIX;  
    const char* filePrefix = "stream-";  
    const char* fileExtension = ".h265";  
  
    DIR* dir = opendir(directoryPath);  
    if (dir == NULL) {  
        perror("opendir");  
        return (void*)-1;  
    }  
  
    struct dirent* entry;  
    while ((entry = readdir(dir)) != NULL) {  
        if (entry->d_type == DT_REG) {  
            const char* fileName = entry->d_name;  
            if (strncmp(fileName, filePrefix, strlen(filePrefix)) == 0 &&  
                strstr(fileName, fileExtension) != NULL) {  
                videoFiles.push_back(std::string(directoryPath) + "/" + fileName);  
            }  
        }  
    }  
    closedir(dir);  
  
    printf("Files to merge:\n");  
    for (const auto& file : videoFiles) {  
        printf("%s\n", file.c_str());  
    }  
  
    std::string ffmpegPath = "./ffmpeg";  
    std::string tempFilePath = "temp_file_list.txt";  
    if (!createTempFileList(videoFiles, tempFilePath)) {  
        return (void*)1;  
    }  
  
    std::string concatenatedFile = "concatenated.h265";  
    std::string concatCmd = ffmpegPath + " -f concat -safe 0 -i " + tempFilePath + " -c copy " + concatenatedFile;  
    printf("Executing: %s\n", concatCmd.c_str());  
    if (system(concatCmd.c_str()) != 0) {  
        fprintf(stderr, "Error concatenating files.\n");  
        return (void*)1;  
    }  
  
    std::time_t t = std::time(nullptr);  
    std::tm* tm = std::localtime(&t);  
    char time_str[100];  
    std::strftime(time_str, sizeof(time_str), "%Y%m%d%H%M%S", tm);  
    std::string filename = "memVideo_" + std::string(time_str) + ".mp4";  
  
    // std::string ffmpegCmd = ffmpegPath + " -v debug -threads 1 -i " + concatenatedFile + " -c:v libx264 -preset ultrafast -crf 22 " + filename;  
	std::string ffmpegCmd = ffmpegPath + " -v error -threads 1 -i " + concatenatedFile + " -c:v libx264 -preset ultrafast -crf 22 " + filename;  

    printf("Executing: %s\n", ffmpegCmd.c_str());  
    int result = system(ffmpegCmd.c_str());  
    if (result != 0) {  
        fprintf(stderr, "Error re-encoding file.\n");  
        return (void*)(long)result;  
    }  
  
    printf("Successfully merged and re-encoded videos\n");  
  
    remove(tempFilePath.c_str());  
    remove(concatenatedFile.c_str());  

	  // Reset n_clips after merge is done  

    // pthread_mutex_lock(&rec_mutex);  
    // nClips = 0;  // This update won't affect the argument passed to merge thread  
    // isMerge = false;  
    // pthread_mutex_unlock(&rec_mutex); 
  
    return (void*)0;  
}  


int main(int argc, char *argv[])
{
	signal(SIGINT, handle_sigint); 
	int i, ret;
	IMPIVSInterface *interface = NULL;
	IMP_IVS_MoveParam param;
	IMP_IVS_MoveOutput *result = NULL;
	IMPFrameInfo frame;
	unsigned char * g_sub_nv12_buf_move = 0;
	chn[0].enable = 0;
	chn[1].enable = 1;
	chn[2].enable = 0;
	chn[3].enable = 0;
	chn[4].enable = 0;
	chn[5].enable = 0;
	chn[6].enable = 0;

	int sensor_sub_width = 640;
	int sensor_sub_height = 360;
	/* Step.1(1) System init */
	ret = sample_system_init();
	if (ret < 0) {
		IMP_LOG_ERR(TAG, "IMP_System_Init() failed\n");
		return -1;
	}
	/* Step.2(2) FrameSource init */
	ret = sample_framesource_init();
	if (ret < 0) {
		IMP_LOG_ERR(TAG, "FrameSource init failed\n");
		return -1;
	}
	printf("framesource init success.\n");


/////////////////////////////////////////////////////////////////////////  For Recording Step.3 ~ 5 //////////////////////////////////////////
	
	/* Step.3 Create encoder group */
	for (i = 0; i < FS_CHN_NUM; i++) {
		if (chn[i].enable) {
			ret = IMP_Encoder_CreateGroup(chn[i].index);
			if (ret < 0) {
				IMP_LOG_ERR(TAG, "IMP_Encoder_CreateGroup(%d) error !\n", i);
				return -1;
			}
		}
	}
	printf("IMP_Encoder_CreateGroup success.\n");


	/* Step.4 Encoder init */
	ret = sample_encoder_init();
	if (ret < 0) {
		IMP_LOG_ERR(TAG, "Encoder init failed\n");
		return -1;
	}
	printf("sample_encoder_init success.\n");

	// ret = sample_jpeg_init();
	// if (ret < 0) {
	// 	IMP_LOG_ERR(TAG, "Encoder init failed\n");
	// 	return -1;
	// }

	/* Step.5 Bind */
	for (i = 0; i < FS_CHN_NUM; i++) {
		if (chn[i].enable) {
			ret = IMP_System_Bind(&chn[i].framesource_chn, &chn[i].imp_encoder);
			if (ret < 0) {
				IMP_LOG_ERR(TAG, "Bind FrameSource channel%d and Encoder failed\n",i);
				return -1;
			}
		}
	}
	printf("IMP_System_Bind success.\n");


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




	g_sub_nv12_buf_move = (unsigned char *)malloc(sensor_sub_width * sensor_sub_height * 3 / 2);
	if (g_sub_nv12_buf_move == 0) {
		printf("error(%s,%d): malloc buf failed \n", __func__, __LINE__);
		return NULL;
	}

	/* Step.6(3) framesource Stream On */
	ret = sample_framesource_streamon();
	if (ret < 0) {
		IMP_LOG_ERR(TAG, "ImpStreamOn failed\n");
		return -1;
	}
	printf("ImpStreamOn success.\n");

	/* Step.7(4) ivs move start */
	ret = sample_venus_init();
	if (ret < 0) {
		IMP_LOG_ERR(TAG, "sample_venus_init failed\n");
		return -1;
	}

	/* Step.8(5) start to get ivs move result */
	int n_clips = 0;
	int NR_FRAMES_TO_REC = 50;
	while (keep_running) {
		ret = IMP_FrameSource_SnapFrame(1, PIX_FMT_NV12, sensor_sub_width, sensor_sub_height, g_sub_nv12_buf_move, &frame);
		if (ret < 0) {
			printf("%d get frame failed try again\n", 0);
			usleep(30*1000);
			continue;
		}

		frame.virAddr = (unsigned int)g_sub_nv12_buf_move;
		int detect_result = Goto_Magik_Detect((char *)frame.virAddr, sensor_sub_width, sensor_sub_height);

		if (detect_result == 1 && !isRec && !isMerge) {  
            pthread_mutex_lock(&rec_mutex);  
            n_clips++;  
            isRec = true;  
            if (pthread_create(&rec_thread, nullptr, recording_thread_func, &n_clips) != 0) {  
                fprintf(stderr, "Failed to create recording thread\n");  
                isRec = false;  
            }  
            pthread_mutex_unlock(&rec_mutex);  
        } else if (detect_result != 1 && isRec) {  
            if (--NR_FRAMES_TO_REC <= 0) {  
                pthread_mutex_lock(&rec_mutex);  
                isRec = false;  
                pthread_mutex_unlock(&rec_mutex);  
                pthread_join(rec_thread, nullptr);  
                NR_FRAMES_TO_REC = 50;  
            }  
        }  
  
        if (detect_result == 3 && !isRec && !isMerge && n_clips > 1) {  
            pthread_mutex_lock(&rec_mutex);  
            isMerge = true;  
            if (pthread_create(&merge_thread, nullptr, merge_recording_thread_func, &n_clips) != 0) {  
                fprintf(stderr, "Failed to create merge recording thread\n");  
                isMerge = false;  
            }  
            pthread_mutex_unlock(&rec_mutex);  
        }  
  
        // Wait for merge thread to complete outside of the detection loop  
        if (isMerge) {  
            pthread_mutex_lock(&rec_mutex);  
            int joinResult = pthread_join(merge_thread, nullptr);  
            if (joinResult != 0) {  
                fprintf(stderr, "Failed to join merge recording thread: %s\n", strerror(joinResult));  
            }  
            isMerge = false;  
            n_clips = 0; // Reset clips count after merge  
            pthread_mutex_unlock(&rec_mutex);  
        }   

	}

    pthread_mutex_lock(&rec_mutex);  
    if (isRec) {  
        isRec = false;  
        pthread_mutex_unlock(&rec_mutex);  
        pthread_join(rec_thread, NULL);  
    }else {  
        pthread_mutex_unlock(&rec_mutex);  
    }  

	pthread_mutex_lock(&rec_mutex);  
    if (isMerge) {  
        isMerge = false;  
        pthread_mutex_unlock(&rec_mutex);  
        pthread_join(merge_thread, NULL);  
    } else {  
        pthread_mutex_unlock(&rec_mutex);  
    }


	free(g_sub_nv12_buf_move);
	/* Step.9(6) ivs move stop */
	ret = sample_venus_deinit();
	if (ret < 0) {
		IMP_LOG_ERR(TAG, "sample_venus_deinit() failed\n");
		return -1;
	}
	/* Step.10(7) Stream Off */
	ret = sample_framesource_streamoff();
	if (ret < 0) {
		IMP_LOG_ERR(TAG, "FrameSource StreamOff failed\n");
		return -1;
	}


////////////////////////////////////////////////////// release after recording ////////////////////////////////////
	for (i = 0; i < FS_CHN_NUM; i++) {
		if (chn[i].enable) {
			ret = IMP_System_UnBind(&chn[i].framesource_chn, &chn[i].imp_encoder);
			if (ret < 0) {
				IMP_LOG_ERR(TAG, "UnBind FrameSource channel%d and Encoder failed\n",i);
				return -1;
			}
		}
	}

	ret = sample_encoder_exit();
	if (ret < 0) {
		IMP_LOG_ERR(TAG, "Encoder exit failed\n");
		return -1;
	}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	/* Step.11(8) FrameSource exit */
	ret = sample_framesource_exit();
	if (ret < 0) {
		IMP_LOG_ERR(TAG, "FrameSource exit failed\n");
		return -1;
	}
	/* Step.12(9) System exit */
	ret = sample_system_exit();
	if (ret < 0) {
		IMP_LOG_ERR(TAG, "sample_system_exit() failed\n");
		return -1;
	}

	pthread_mutex_destroy(&rec_mutex); 
	return 0;
}
