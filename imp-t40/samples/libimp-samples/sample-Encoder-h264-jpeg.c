/*
 * sample-Encoder-h264-jpeg.c
 *
 * Copyright (C) 2014 Ingenic Semiconductor Co.,Ltd
 *
 * The specific description of all APIs called in this file can be viewed in the header file in the proj/sdk-lv3/include/api/cn/imp/ directory
 *
 * Step.1 System init System initialization
 *		@code
 *			memset(&sensor_info, 0, sizeof(sensor_info));
 *			if(SENSOR_NUM == IMPISP_TOTAL_ONE){
 *				memcpy(&sensor_info[0], &Def_Sensor_Info[0], sizeof(IMPSensorInfo));
 *			} else if(SENSOR_NUM == IMPISP_TOTAL_TWO){
 *				memcpy(&sensor_info[0], &Def_Sensor_Info[0], sizeof(IMPSensorInfo) * 2);
 *			}else if(SENSOR_NUM ==IMPISP_TOTAL_THR){
 *				memcpy(&sensor_info[0], &Def_Sensor_Info[0], sizeof(IMPSensorInfo) * 3)
 *			} //According to the number of sensors, copied the contents of the corresponding size Def_Sensor_Info to sensor_info
 *
 *			ret = IMP_ISP_Open() //Open the ISP module
 *			ret = IMP_ISP_SetCameraInputMode(&mode) //If there are multiple sensors (maximum support of three cameras), set to multiple cameras mode(please ignore if you're using single camera)
 *			ret = IMP_ISP_AddSensor(IMPVI_MAIN, &sensor_info[*]) //Add sensor, before this operation the sensor driver has been added to the kernel (IMPVI_MAIN is the main camera, IMPVI_SEC is the second camera, IMPVI_THR is the third camera)
 *			ret = IMP_ISP_EnableSensor(IMPVI_MAIN, &sensor_info[*])	//Enable sensor, Now the sensor starts outputting the image (IMPVI_MAIN is the main camera, IMPVI_SEC is the second camera, IMPVI_THR is the third camera)
 *			ret = IMP_System_Init() //System initialization
 *			ret = IMP_ISP_EnableTuning() //Enable ISP tuning before calling the ISP debugging interface
 *		@endcode
 * Step.2 FrameSource init Framesource initialization
 *		@code
 *			ret = IMP_FrameSource_CreateChn(chn[i].index, &chn[i].fs_chn_attr) //Create channel
 *			ret = IMP_FrameSource_SetChnAttr(chn[i].index, &chn[i].fs_chn_attr) //Set channel-related properties, including: image width, image height, image format, output frame rate of the channel, number of cache buf, cropping and scaling properties
 *		@endcode
 * Step.3 Encoder init Encoding initialization
 *		@code
 *			ret = IMP_Encoder_CreateGroup(chn[i].index) //Create encoding group
 *			ret = sample_encoder_init() //Video encoding initialization, the specific implementation can refer to the comments in sample-encoder-video.c
 *			ret = sample_jpeg_init() //Image encoding initialization, the specific implementation can refer to the comments in sample-encoder-jpeg.c
 *		@endcode
 * Step.4 Bind    Bind framesource and encode chnnel
 *		@code
 *			ret = IMP_System_Bind(&chn[i].framesource_chn, &chn[i].imp_encoder)	//Bind the framesource and the encoded chnnel, and the data generated by the framesource can be automatically transmitted to the encoded chnnel after the binding is successful
 *		@endcode
 * Step.5 Stream On Enable Framesource chnnel, start outputting the image
 *		@code
 *			ret = IMP_FrameSource_EnableChn(chn[i].index) //Eable chnnel, chnnel starts to output image
 *		@endcode
 * Step.6 Get stream and Snap   Get bitstream and JPEG-encoded images
 *		@code
 *			ret = pthread_create(&tid, NULL, h264_stream_thread, NULL) //Use another thread to get and save the bitstream, the specific implementation can refer to the comments in sample-encoder-video.c
 *			ret = sample_get_jpeg_snap() //Obtain the code stream and save the encoded picture, please refer to the comments in sample-encoder-jpeg.c for the specific implementation
 *		@endcode
 * Step.7 Stream Off   Disable Framesource chnnel, stop image output
 *		@code
 *			ret = IMP_FrameSource_DisableChn(chn[i].index) //Disable channel, channel stop stops outputting images
 *		@endcode
 * Step.8 UnBind Unbind Framesource and code chnnel
 *		@code
 *			ret = IMP_System_UnBind(&chn[i].framesource_chn, &chn[i].imp_encoder) //Unbind Framesource and code chnnel
 *		@endcode
 * Step.9 Encoder exit  Encode deinitialization
 *		@code
 *			ret = sample_jpeg_exit() //Image encoding deinitialization, the specific implementation can refer to comments in sample-encoder-jpeg.c 
 *			ret = sample_encoder_exit() //Video encoding deinitialization, the specific implementation can refer to the comments in sample-encoder-video.c
 *		@endcode
 * Step.10 FrameSource exit Framesource deinitialization
 *		@code
 *			ret = IMP_FrameSource_DestroyChn(chn[i].index) //Destory channel
 *		@endcode
 * Step.11 System exit System deinitialization
 *		@code
 *			ret = IMP_ISP_DisableTuning() //Disable ISP tuning
 *			ret = IMP_System_Exit() //System deinitialization
 *			ret = IMP_ISP_DisableSensor(IMPVI_MAIN, &sensor_info[*]) //Disable sensor, sensor stop outputting image (IMPVI_MAIN in the main camera, IMPVI_SEC is the second camera, IMPVI_THR is the third camera)
 *			ret = IMP_ISP_DelSensor(IMPVI_MAIN, &sensor_info[*]) //Delete sensor (IMPVI_MAIN is main camera, IMPVI_SEC is second camera, IMPVI_THR is third camera)
 *			ret = IMP_ISP_Close() //Turn off ISP module
 *		@endcode
 * */
#include <pthread.h>
#include <stdio.h>
#include <string.h>

#include <imp/imp_log.h>
#include <imp/imp_common.h>
#include <imp/imp_system.h>
#include <imp/imp_framesource.h>
#include <imp/imp_encoder.h>

#include "sample-common.h"

#define TAG "Sample-Encoder-h264-jpeg"

extern struct chn_conf chn[];

static void *h264_stream_thread(void *m)
{
	int ret;
	ret = sample_get_video_stream();
	if (ret < 0) {
		IMP_LOG_ERR(TAG, "Get H264 stream failed\n");
		return 0;
	}

	return 0;
}

int main(int argc, char *argv[])
{
	int i, ret;
	chn[1].enable = 0;
	chn[2].enable = 0;

	/* Step.1 System init */
	ret = sample_system_init();
	if (ret < 0) {
		IMP_LOG_ERR(TAG, "IMP_System_Init() failed\n");
		return -1;
	}

	/* Step.2 FrameSource init */
	ret = sample_framesource_init();
	if (ret < 0) {
		IMP_LOG_ERR(TAG, "FrameSource init failed\n");
		return -1;
	}

	/* Step.3 Encoder init */
	for (i = 0; i < FS_CHN_NUM; i++) {
		if (chn[i].enable) {
			ret = IMP_Encoder_CreateGroup(chn[i].index);
			if (ret < 0) {
				IMP_LOG_ERR(TAG, "IMP_Encoder_CreateGroup(%d) error !\n", i);
				return -1;
			}
		}
	}

	ret = sample_encoder_init();
	if (ret < 0) {
		IMP_LOG_ERR(TAG, "Encoder init failed\n");
		return -1;
	}

	ret = sample_jpeg_init();
	if (ret < 0) {
		IMP_LOG_ERR(TAG, "Encoder init failed\n");
		return -1;
	}

	/* Step.4 Bind */
	for (i = 0; i < FS_CHN_NUM; i++) {
		if (chn[i].enable) {
			ret = IMP_System_Bind(&chn[i].framesource_chn, &chn[i].imp_encoder);
			if (ret < 0) {
				IMP_LOG_ERR(TAG, "Bind FrameSource channel%d and Encoder failed\n",i);
				return -1;
			}
		}
	}

	/* Step.5 Stream On */
	ret = sample_framesource_streamon();
	if (ret < 0) {
		IMP_LOG_ERR(TAG, "ImpStreamOn failed\n");
		return -1;
	}

	/* Step.6 Get stream and Snap */
	pthread_t tid; /* Stream capture in another thread */
	ret = pthread_create(&tid, NULL, h264_stream_thread, NULL);
	if (ret) {
		IMP_LOG_ERR(TAG, "h264 stream create error\n");
		return -1;
	}

	/* drop several pictures of invalid data */
	sleep(SLEEP_TIME);

	ret = sample_get_jpeg_snap();
	if (ret < 0) {
		IMP_LOG_ERR(TAG, "Get H264 stream failed\n");
		return -1;
	}

	pthread_join(tid, NULL);

	/* Exit sequence as follow */
	/* Step.7 Stream Off */
	ret = sample_framesource_streamoff();
	if (ret < 0) {
		IMP_LOG_ERR(TAG, "FrameSource StreamOff failed\n");
		return -1;
	}

	/* Step.8 UnBind */
	for (i = 0; i < FS_CHN_NUM; i++) {
		if (chn[i].enable) {
			ret = IMP_System_UnBind(&chn[i].framesource_chn, &chn[i].imp_encoder);
			if (ret < 0) {
				IMP_LOG_ERR(TAG, "UnBind FrameSource channel%d and Encoder failed\n",i);
				return -1;
			}
		}
	}

	/* Step.9 Encoder exit */
	ret = sample_jpeg_exit();
	if (ret < 0) {
		IMP_LOG_ERR(TAG, "Encoder jpeg exit failed\n");
		return -1;
	}

	ret = sample_encoder_exit();
	if (ret < 0) {
		IMP_LOG_ERR(TAG, "Encoder exit failed\n");
		return -1;
	}

	/* Step.10 FrameSource exit */
	ret = sample_framesource_exit();
	if (ret < 0) {
		IMP_LOG_ERR(TAG, "FrameSource exit failed\n");
		return -1;
	}

	/* Step.11 System exit */
	ret = sample_system_exit();
	if (ret < 0) {
		IMP_LOG_ERR(TAG, "sample_system_exit() failed\n");
		return -1;
	}

	return 0;
}
