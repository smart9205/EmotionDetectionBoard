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

#define FACE_MODEL_PATH "face.bin"
#define EMO_MODEL_PATH "emo.bin"

extern struct chn_conf chn[];
std::unique_ptr<venus::BaseNet> face_net;
std::unique_ptr<venus::BaseNet> emo_net;

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

int main(int argc, char *argv[])
{
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

	// chn[1].enable = 0;
	// chn[2].enable = 0;

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
	bool isRec = false;
	int n_clips = 0;
	int rec_start_time = 0;
	int rec_stop_time = 0;
	for (i = 0; i < 3*NR_FRAMES_TO_SAVE; i++) {

		ret = IMP_FrameSource_SnapFrame(1, PIX_FMT_NV12, sensor_sub_width, sensor_sub_height, g_sub_nv12_buf_move, &frame);
		if (ret < 0) {
			printf("%d get frame failed try again\n", 0);
			usleep(30*1000);
		}
		frame.virAddr = (unsigned int)g_sub_nv12_buf_move;
		ret = Goto_Magik_Detect((char *)frame.virAddr, sensor_sub_width, sensor_sub_height);

		if (ret == 1 && !isRec){
			n_clips ++;
			printf("Recording started........\n");
			isRec = true;
			int recstate = sample_get_video_stream(n_clips, 0);
			if (recstate < 0) {
				IMP_LOG_ERR(TAG, "Get H264 stream failed\n");
				printf("Get H264 stream failed\n");
				return -1;
			}
			rec_start_time = i;
		}
		if (ret != 1 && isRec && (i-rec_start_time) > NR_FRAMES_TO_REC){
			int recstate = sample_get_video_stream(n_clips, 1);
			isRec = false;
			rec_stop_time = i;
			printf("stopping recording........\n");
		}
	}
	
	if (isRec)
		int recstate = sample_get_video_stream(n_clips, 1);

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
	return 0;
}
