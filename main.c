  #include <stdlib.h>
  #include <stdio.h>
  #include <opencv\cv.h>
  #include <opencv\highgui.h>
  #include <GL/glut.h>
  #include <libfreenect_sync.h>
  #include "libfidtrack\fidtrackX.h"
  #include "libfidtrack\segment.h"

  #define PI 3.14159265


	double angle( CvPoint* pt1, CvPoint* pt2, CvPoint* pt0 )
	{
		double dx1 = pt1->x - pt0->x;
		double dy1 = pt1->y - pt0->y;
		double dx2 = pt2->x - pt0->x;
		double dy2 = pt2->y - pt0->y;
		return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
	}

	int compare_square_points_y(const void *a, const void *b) 
	{ 
		CvPoint2D32f *ia = (CvPoint2D32f*)a;
		CvPoint2D32f *ib = (CvPoint2D32f*)b;
		return (int)(ia->y - ib->y);
	}

	int compare_square_points_x(const void *a, const void *b) 
	{ 
		CvPoint2D32f *ia = (CvPoint2D32f*)a;
		CvPoint2D32f *ib = (CvPoint2D32f*)b;
		return (int)(ia->x - ib->x);
	}

	void sortTagSquares(CvPoint2D32f *point_array)
	{
		CvPoint2D32f first3[3];
		CvPoint2D32f last3[3];

		qsort(point_array, 10, sizeof(CvPoint2D32f), compare_square_points_y);

		first3[0] = point_array[0];
		first3[1] = point_array[1];
		first3[2] = point_array[2];
		last3[0] = point_array[7];
		last3[1] = point_array[8];
		last3[2] = point_array[9];

		qsort(first3, 3, sizeof(CvPoint2D32f), compare_square_points_x);
		point_array[0] = first3[0];
		point_array[1] = first3[1];
		point_array[2] = first3[2];

		qsort(last3, 3, sizeof(CvPoint2D32f), compare_square_points_x);
		point_array[7] = last3[0];
		point_array[8] = last3[1];
		point_array[9] = last3[2];
	}

	void sortTagSquaresAligned(CvPoint2D32f *point_array)
	{
		CvPoint2D32f first3[3];
		CvPoint2D32f middle2[2];
		CvPoint2D32f last3[3];

		qsort(point_array, 8, sizeof(CvPoint2D32f), compare_square_points_y);

		first3[0] = point_array[0];
		first3[1] = point_array[1];
		first3[2] = point_array[2];
		middle2[0] = point_array[3];
		middle2[1] = point_array[4];
		last3[0] = point_array[5];
		last3[1] = point_array[6];
		last3[2] = point_array[7];

		qsort(first3, 3, sizeof(CvPoint2D32f), compare_square_points_x);
		point_array[0] = first3[0];
		point_array[1] = first3[1];
		point_array[2] = first3[2];

		qsort(middle2, 2, sizeof(CvPoint2D32f), compare_square_points_x);
		point_array[3] = middle2[0];
		point_array[4] = middle2[1];

		qsort(last3, 3, sizeof(CvPoint2D32f), compare_square_points_x);
		point_array[5] = last3[0];
		point_array[6] = last3[1];
		point_array[7] = last3[2];
	}

  int main()
  {
		IplImage *image = cvCreateImageHeader(cvSize(640,480), 8, 3);
		IplImage *imgTest = cvCreateImage(cvSize(640,480), 8, 3);
		IplImage *imgHSV = cvCreateImage(cvSize(640,480), 8, 3);
        IplImage *imgThreshed = cvCreateImage(cvSize(640,480), 8, 1);
		IplImage *imgBW = cvCreateImage(cvSize(640,480),IPL_DEPTH_8U,1);
		//IplImage *imgBWC = cvCreateImage(cvSize(640,480),IPL_DEPTH_8U,1);
		IplImage *imgCont = cvCreateImage(cvSize(640,480),IPL_DEPTH_8U,1);
		

		/*
		IplImage *depthImage = cvCreateImage(cvSize(640,480), 16, 1);
		*/

		CvMemStorage *storage = cvCreateMemStorage(0);
		CvMemStorage *contStorage = cvCreateMemStorage(0);
		CvSeq *contours = 0;
		CvSeq *square = 0;
		schar *points = 0;

		//fiducial tracking
		Segmenter segmenter;
		TreeIdMap treeidmap;

		CvSeq *result;
		CvSeq* squares = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvPoint), storage );
		CvSeqReader reader;
		CvPoint pt[4];

		//chessboard calibration test
		const int corner_cnt_x = 10;
		const int corner_cnt_y = 7;
		const int corner_cnt = corner_cnt_x*corner_cnt_y;
		const float square_size = 2.5;
		CvPoint2D32f corners_rgb[70];
		int corner_count;
		int calibrated = 0;

		CvMat* object_points = cvCreateMat(70, 3, CV_32FC1);
		CvMat* image_points_rgb = cvCreateMat(70, 2, CV_32FC1);
		CvMat* point_counts = cvCreateMat(1, 1, CV_32SC1);
		CvMat* camera_matrix_rgb = cvCreateMat(3, 3, CV_32FC1);
		CvMat* distortion_coeffs_rgb = cvCreateMat(5, 1, CV_32FC1);
		
		CvMat *rotation_vector = cvCreateMat(1, 3, CV_32FC1);
		CvMat *translation_vector = cvCreateMat(1, 3, CV_32FC1);
		CvMat *rotation_matrix = cvCreateMat(3, 3, CV_32FC1);
		CvMat *rotation_translation_matrix = cvCreateMat(3,4,CV_32FC1);
		CvMat *projection_matrix;

		CvMat* modelPoints = cvCreateMat(8, 3, CV_32FC1);
		CvMat* realPoints = cvCreateMat(8, 3, CV_32FC1);

		//3d cube on tag
		//non-aligned tag 10
		//CvPoint2D32f squares_tag[10];
		//CvMat* object_points_tag = cvCreateMat(10, 3, CV_32FC1);
		//CvMat* image_points_tag = cvCreateMat(10, 2, CV_32FC1);
		
		//aligned tag 8
		CvPoint2D32f squares_tag[8];
		CvMat* object_points_tag = cvCreateMat(8, 3, CV_32FC1);
		CvMat* image_points_tag = cvCreateMat(8, 2, CV_32FC1);

		CvMat* point_counts_tag = cvCreateMat(1, 1, CV_32SC1);
		int total_squares = 0;

		FidtrackerX fidtrackerx;
		FiducialX fiducials[ 128 ];
		RegionX regions[ 128 * 2 ];
		int fidCount;
		const unsigned char* pixels;

		initialize_treeidmap_from_file(&treeidmap, "all.trees");
		initialize_fidtrackerX( &fidtrackerx, &treeidmap, NULL );
		initialize_segmenter( &segmenter, 640, 480, treeidmap.max_adjacencies );
		
		
		//PI = atan(1.0)*4;
		while (cvWaitKey(10) < 0) 
		{
			//const unsigned char* pixels;
			int i;
			double s, t;
			char *data;
			char *depth;
			unsigned int timestamp;
			unsigned int timestamp2;
			freenect_sync_set_led(LED_GREEN, 0);
			freenect_sync_get_video((void**)(&data), &timestamp, 0, FREENECT_VIDEO_RGB);
			//freenect_sync_get_depth((void**)(&depth), &timestamp2, 0, FREENECT_DEPTH_11BIT);

			//setup depth image
			//cvSetData(depthImage, depth, depthImage->widthStep);
			//cvReleaseImageHeader(&depthImage);

			//setup rgb image
			cvSetData(image, data, 640*3);
			cvCvtColor(image, image, CV_RGB2BGR);
			cvCvtColor(image, imgHSV, CV_BGR2HSV);
			cvCvtColor(image, imgBW, CV_BGR2GRAY);
			//cvThreshold(imgBW, imgBW, 128, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
			//cvThreshold(imgBW, imgBWC, 128, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
			cvThreshold(imgBW, imgBW, 128, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
			

			//run-once calibration
			if(calibrated == 0)
			{
				corner_count = 0;
				if (cvFindChessboardCorners(imgBW, cvSize(10, 7), corners_rgb, &corner_count, 3) != 0 && corner_count == 70)
				{
					cvFindCornerSubPix(imgBW, corners_rgb, corner_count, cvSize(7,7), cvSize(-1,-1), cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1));
					
					for (i = 0; i<corner_cnt; i++)
					{
						cvSetReal2D(object_points, i, 0, i%10 *2.5);
						cvSetReal2D(object_points, i, 1, i/10 *2.5);
						cvSetReal2D(object_points, i, 2, 0.0);
						printf("i: %d x: %f, y: %f\n", i, i%10*2.5, i/10*2.5);

						cvSetReal2D(image_points_rgb, i, 0, corners_rgb[i].x);
						cvSetReal2D(image_points_rgb, i, 1, corners_rgb[i].y);
					}


					cvSetReal1D(point_counts, 0, (double) corner_count);
					cvSet1D(distortion_coeffs_rgb, 4, cvScalarAll(0));
					
					cvCalibrateCamera2(object_points, image_points_rgb, point_counts, cvSize(640,480), camera_matrix_rgb, distortion_coeffs_rgb, NULL, NULL, CV_CALIB_FIX_K3);
					cvFindExtrinsicCameraParams2(object_points, image_points_rgb, camera_matrix_rgb, distortion_coeffs_rgb, rotation_vector, translation_vector, 0);
					printf("translation vector: %f, %f, %f\n", cvGetReal2D(translation_vector, 0, 0),cvGetReal2D(translation_vector, 0, 1),cvGetReal2D(translation_vector, 0, 2) );
					printf("rotation vector vector: %f, %f, %f\n", cvGetReal2D(rotation_vector, 0, 0),cvGetReal2D(rotation_vector, 0, 1),cvGetReal2D(rotation_vector, 0, 2) );
					cvRodrigues2(rotation_vector, rotation_matrix, 0);
					
					
					//3d cube
					cvSetReal2D(modelPoints, 0, 0, 0.0f);
					cvSetReal2D(modelPoints, 0, 1, 0.0f);
					cvSetReal2D(modelPoints, 0, 2, 0.0f);

					cvSetReal2D(modelPoints, 1, 0, 10.0f);
					cvSetReal2D(modelPoints, 1, 1, 0.0f);
					cvSetReal2D(modelPoints, 1, 2, 0.0f);

					cvSetReal2D(modelPoints, 2, 0, 10.0f);
					cvSetReal2D(modelPoints, 2, 1, 10.0f);
					cvSetReal2D(modelPoints, 2, 2, 0.0f);

					cvSetReal2D(modelPoints, 3, 0, 0.0f);
					cvSetReal2D(modelPoints, 3, 1, 10.0f);
					cvSetReal2D(modelPoints, 3, 2, 0.0f);

					cvSetReal2D(modelPoints, 4, 0, 0.0f);
					cvSetReal2D(modelPoints, 4, 1, 0.0f);
					cvSetReal2D(modelPoints, 4, 2, -10.0f);

					cvSetReal2D(modelPoints, 5, 0, 10.0f);
					cvSetReal2D(modelPoints, 5, 1, 0.0f);
					cvSetReal2D(modelPoints, 5, 2, -10.0f);

					cvSetReal2D(modelPoints, 6, 0, 10.0f);
					cvSetReal2D(modelPoints, 6, 1, 10.0f);
					cvSetReal2D(modelPoints, 6, 2, -10.0f);

					cvSetReal2D(modelPoints, 7, 0, 0.0f);
					cvSetReal2D(modelPoints, 7, 1, 10.0f);
					cvSetReal2D(modelPoints, 7, 2, -10.0f);

					cvSave("Intrinsics_RGB.xml",camera_matrix_rgb, 0, 0, cvAttrList(0, 0));
					cvSave("Distortion_RGB.xml",distortion_coeffs_rgb, 0, 0, cvAttrList(0, 0));
					
					cvDrawChessboardCorners(image, cvSize(10, 7), corners_rgb, corner_count, 1);
					cvShowImage("RGB", image);
					printf("Calibrated. %d corners.\n", corner_cnt);
					//cvWaitKey(10000);
					calibrated = 1;
				}
				printf("Please display the x10y7 cornered chessboard pattern to the camera for calibration\n");
				cvWaitKey(1000);
			}
			else 
			{
				//draw cube using the chessboard as the "tag" instead of my custom mixed tag

				/*
				corner_count = 0;
				if (cvFindChessboardCorners(imgBW, cvSize(10, 7), corners_rgb, &corner_count, 3) != 0 && corner_count == 70)
				{
					cvFindCornerSubPix(imgBW, corners_rgb, corner_count, cvSize(7,7), cvSize(-1,-1), cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1));
					
					for (i = 0; i<corner_cnt; i++)
					{
						cvSetReal2D(object_points, i, 0, i%10 *2.5);
						cvSetReal2D(object_points, i, 1, i/10 *2.5);
						cvSetReal2D(object_points, i, 2, 0.0);

						cvSetReal2D(image_points_rgb, i, 0, corners_rgb[i].x);
						cvSetReal2D(image_points_rgb, i, 1, corners_rgb[i].y);
					}
				
				cvSetReal1D(point_counts, 0, (double) corner_count);
				cvFindExtrinsicCameraParams2(object_points, image_points_rgb, camera_matrix_rgb, distortion_coeffs_rgb, rotation_vector, translation_vector, 0);
				cvProjectPoints2(modelPoints, rotation_vector, translation_vector, camera_matrix_rgb, distortion_coeffs_rgb, realPoints, 0, 0, 0, 0, 0, 640/480);

				//draw
				cvProjectPoints2(modelPoints, rotation_vector, translation_vector, camera_matrix_rgb, distortion_coeffs_rgb, realPoints, 0, 0, 0, 0, 0, 0);
				cvDrawLine(image, cvPoint(cvGetReal2D(realPoints, 0, 0), cvGetReal2D(realPoints, 0, 1)), cvPoint(cvGetReal2D(realPoints, 1, 0), cvGetReal2D(realPoints, 1, 1)), CV_RGB(255, 0, 255), 3, 8, 0);
				cvDrawLine(image, cvPoint(cvGetReal2D(realPoints, 1, 0), cvGetReal2D(realPoints, 1, 1)), cvPoint(cvGetReal2D(realPoints, 2, 0), cvGetReal2D(realPoints, 2, 1)), CV_RGB(255, 0, 255), 3, 8, 0);
				cvDrawLine(image, cvPoint(cvGetReal2D(realPoints, 2, 0), cvGetReal2D(realPoints, 2, 1)), cvPoint(cvGetReal2D(realPoints, 3, 0), cvGetReal2D(realPoints, 3, 1)), CV_RGB(255, 0, 255), 3, 8, 0);
				cvDrawLine(image, cvPoint(cvGetReal2D(realPoints, 3, 0), cvGetReal2D(realPoints, 3, 1)), cvPoint(cvGetReal2D(realPoints, 0, 0), cvGetReal2D(realPoints, 0, 1)), CV_RGB(255, 0, 255), 3, 8, 0);

				cvDrawLine(image, cvPoint(cvGetReal2D(realPoints, 4, 0), cvGetReal2D(realPoints, 4, 1)), cvPoint(cvGetReal2D(realPoints, 5, 0), cvGetReal2D(realPoints, 5, 1)), CV_RGB(255, 0, 255), 3, 8, 0);
				cvDrawLine(image, cvPoint(cvGetReal2D(realPoints, 5, 0), cvGetReal2D(realPoints, 5, 1)), cvPoint(cvGetReal2D(realPoints, 6, 0), cvGetReal2D(realPoints, 6, 1)), CV_RGB(255, 0, 255), 3, 8, 0);
				cvDrawLine(image, cvPoint(cvGetReal2D(realPoints, 6, 0), cvGetReal2D(realPoints, 6, 1)), cvPoint(cvGetReal2D(realPoints, 7, 0), cvGetReal2D(realPoints, 7, 1)), CV_RGB(255, 0, 255), 3, 8, 0);
				cvDrawLine(image, cvPoint(cvGetReal2D(realPoints, 7, 0), cvGetReal2D(realPoints, 7, 1)), cvPoint(cvGetReal2D(realPoints, 4, 0), cvGetReal2D(realPoints, 4, 1)), CV_RGB(255, 0, 255), 3, 8, 0);

				cvDrawLine(image, cvPoint(cvGetReal2D(realPoints, 0, 0), cvGetReal2D(realPoints, 0, 1)), cvPoint(cvGetReal2D(realPoints, 4, 0), cvGetReal2D(realPoints, 4, 1)), CV_RGB(255, 0, 255), 3, 8, 0);
				cvDrawLine(image, cvPoint(cvGetReal2D(realPoints, 1, 0), cvGetReal2D(realPoints, 1, 1)), cvPoint(cvGetReal2D(realPoints, 5, 0), cvGetReal2D(realPoints, 5, 1)), CV_RGB(255, 0, 255), 3, 8, 0);
				cvDrawLine(image, cvPoint(cvGetReal2D(realPoints, 2, 0), cvGetReal2D(realPoints, 2, 1)), cvPoint(cvGetReal2D(realPoints, 6, 0), cvGetReal2D(realPoints, 6, 1)), CV_RGB(255, 0, 255), 3, 8, 0);
				cvDrawLine(image, cvPoint(cvGetReal2D(realPoints, 3, 0), cvGetReal2D(realPoints, 3, 1)), cvPoint(cvGetReal2D(realPoints, 7, 0), cvGetReal2D(realPoints, 7, 1)), CV_RGB(255, 0, 255), 3, 8, 0);
				}
				*/
			}


			//FIDUCIAL TRACKING
			pixels = (const unsigned char*) imgBW->imageData;
			step_segmenter( &segmenter, pixels);
			fidCount = find_fiducialsX( fiducials, 128,  &fidtrackerx , &segmenter, 640, 480);
			system("cls");
			printf("Fiducials: %d\n", fidCount);
			for(i = 0; i< fidCount; i++) {
				float angle, size, x, y, dx, dy, hyp, rad;
				printf("%d\n", fiducials[i].id);
				cvCircle(image, cvPoint(fiducials[i].x, fiducials[i].y), 10, cvScalar(0, 0, 255, 0), 1, 8, 0);

			}

			//This is only valid in 2D.
			//cvPoint((hyp * cos(rad + 2.35619449)) + x, (hyp * sin(rad + 2.35619449)) + y) //origin
			//cvPoint((hyp * cos(rad +  0.785398163)) + x, (hyp * sin(rad +  0.785398163)) + y) //x-axis
			//cvPoint((hyp * cos(rad + 5.49778714)) + x, (hyp * sin(rad + 5.49778714)) + y) //3rd square point
			//cvPoint((hyp * cos(rad + 3.92699082)) + x,(hyp * sin(rad + 3.92699082)) + y) //z-axis (OR y-axis)


			//filter the blue
			//cvInRangeS(imgHSV, cvScalar(100, 100, 100, 0), cvScalar(150, 255, 255, 0), imgCont); //blue
			cvInRangeS(imgHSV, cvScalar(100, 100, 30, 0), cvScalar(150, 255, 190, 0), imgCont); //blue REAL LIGHTING CONDITIONS PRINT
			cvFindContours(imgCont, storage, &contours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));
			//cvFindContours(imgBWC, storage, &contours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));
			//cvDrawContours(image, contours, CV_RGB(255,0,0), CV_RGB(0,255,0), 10, 1, CV_AA, cvPoint(0,0));
			
			//FIND THE WEE MAGICAL SQUARES LADDY
			//(inspired by the opencv c++ examples)
			while( contours )
	        {
				result = cvApproxPoly( contours, sizeof(CvContour), storage, CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0 );

	            if( result->total == 4 && fabs(cvContourArea(result,CV_WHOLE_SEQ, 0)) > 150 && cvCheckContourConvexity(result) )
	            {
	                s = 0;
	                   
                    for( i = 0; i < 5; i++ )
	                {
	                    if( i >= 2 )
	                    {
	                        t = fabs(angle( (CvPoint*)cvGetSeqElem( result, i), (CvPoint*)cvGetSeqElem( result, i-2), (CvPoint*)cvGetSeqElem( result, i-1)));
	                        s = s > t ? s : t;
	                    }
	                }
	                   
	                if( s < 0.3 )
	                    for( i = 0; i < 4; i++ )
                        cvSeqPush( squares,
                            (CvPoint*)cvGetSeqElem( result, i));
	            }
	               
                contours = contours->h_next;
	        }


	    cvStartReadSeq( squares, &reader, 0 );
	   
		total_squares = 0;

	    for( i = 0; i < squares->total; i += 4 )
	    {
	        CvPoint* rect = pt;
	        int count = 4;
	       
	        memcpy( pt, reader.ptr, squares->elem_size );
	        CV_NEXT_SEQ_ELEM( squares->elem_size, reader );
	        memcpy( pt + 1, reader.ptr, squares->elem_size );
	        CV_NEXT_SEQ_ELEM( squares->elem_size, reader );
	        memcpy( pt + 2, reader.ptr, squares->elem_size );
	        CV_NEXT_SEQ_ELEM( squares->elem_size, reader );
	        memcpy( pt + 3, reader.ptr, squares->elem_size );
	        CV_NEXT_SEQ_ELEM( squares->elem_size, reader );
	       

	        cvPolyLine( image, &rect, &count, 1, 1, CV_RGB(0,255,0), 3, 8, 0 );
			cvDrawCircle( image, cvPoint((rect[0].x+rect[1].x+rect[2].x+rect[3].x)/4, (rect[0].y+rect[1].y+rect[2].y+rect[3].y)/4), 2, CV_RGB(255, 0, 0), 1, 8, 0);
			squares_tag[total_squares] = cvPoint2D32f((rect[0].x+rect[1].x+rect[2].x+rect[3].x)/4, (rect[0].y+rect[1].y+rect[2].y+rect[3].y)/4);
			total_squares++;

	    }
		squares = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvPoint), storage );

		//sort the points and create point and 3D representation lists
		
		if(total_squares == 8) {

			//aligned tag
			sortTagSquaresAligned(squares_tag);
			cvSetReal2D(object_points_tag, 0, 0, 0);
			cvSetReal2D(object_points_tag, 0, 1, 0);
			cvSetReal2D(object_points_tag, 0, 2, 0.0);
			cvSetReal2D(image_points_tag, 0, 0, squares_tag[0].x);
			cvSetReal2D(image_points_tag, 0, 1, squares_tag[0].y);

			cvSetReal2D(object_points_tag, 1, 0, 5);
			cvSetReal2D(object_points_tag, 1, 1, 0);
			cvSetReal2D(object_points_tag, 1, 2, 0.0);
			cvSetReal2D(image_points_tag, 1, 0, squares_tag[1].x);
			cvSetReal2D(image_points_tag, 1, 1, squares_tag[1].y);

			cvSetReal2D(object_points_tag, 2, 0, 10);
			cvSetReal2D(object_points_tag, 2, 1, 0);
			cvSetReal2D(object_points_tag, 2, 2, 0.0);
			cvSetReal2D(image_points_tag, 2, 0, squares_tag[2].x);
			cvSetReal2D(image_points_tag, 2, 1, squares_tag[2].y);

			cvSetReal2D(object_points_tag, 3, 0, 0);
			cvSetReal2D(object_points_tag, 3, 1, 5);
			cvSetReal2D(object_points_tag, 3, 2, 0.0);
			cvSetReal2D(image_points_tag, 3, 0, squares_tag[3].x);
			cvSetReal2D(image_points_tag, 3, 1, squares_tag[3].y);

			cvSetReal2D(object_points_tag, 4, 0, 10);
			cvSetReal2D(object_points_tag, 4, 1, 5);
			cvSetReal2D(object_points_tag, 4, 2, 0.0);
			cvSetReal2D(image_points_tag, 4, 0, squares_tag[4].x);
			cvSetReal2D(image_points_tag, 4, 1, squares_tag[4].y);

			cvSetReal2D(object_points_tag, 5, 0, 0);
			cvSetReal2D(object_points_tag, 5, 1, 10);
			cvSetReal2D(object_points_tag, 5, 2, 0.0);
			cvSetReal2D(image_points_tag, 5, 0, squares_tag[5].x);
			cvSetReal2D(image_points_tag, 5, 1, squares_tag[5].y);

			cvSetReal2D(object_points_tag, 6, 0, 5);
			cvSetReal2D(object_points_tag, 6, 1, 10);
			cvSetReal2D(object_points_tag, 6, 2, 0.0);
			cvSetReal2D(image_points_tag, 6, 0, squares_tag[6].x);
			cvSetReal2D(image_points_tag, 6, 1, squares_tag[6].y);

			cvSetReal2D(object_points_tag, 7, 0, 10);
			cvSetReal2D(object_points_tag, 7, 1, 10);
			cvSetReal2D(object_points_tag, 7, 2, 0.0);
			cvSetReal2D(image_points_tag, 7, 0, squares_tag[7].x);
			cvSetReal2D(image_points_tag, 7, 1, squares_tag[7].y);

			//find rotation and translation vectors and project points from 3D to 2D pixel coordinates
			cvFindExtrinsicCameraParams2(object_points_tag, image_points_tag, camera_matrix_rgb, distortion_coeffs_rgb, rotation_vector, translation_vector, 0);
			cvProjectPoints2(modelPoints, rotation_vector, translation_vector, camera_matrix_rgb, distortion_coeffs_rgb, realPoints, 0, 0, 0, 0, 0, 640/480);

			cvDrawLine(image, cvPoint(cvGetReal2D(realPoints, 0, 0), cvGetReal2D(realPoints, 0, 1)), cvPoint(cvGetReal2D(realPoints, 1, 0), cvGetReal2D(realPoints, 1, 1)), CV_RGB(255, 0, 255), 3, 8, 0);
			cvDrawLine(image, cvPoint(cvGetReal2D(realPoints, 1, 0), cvGetReal2D(realPoints, 1, 1)), cvPoint(cvGetReal2D(realPoints, 2, 0), cvGetReal2D(realPoints, 2, 1)), CV_RGB(255, 0, 255), 3, 8, 0);
			cvDrawLine(image, cvPoint(cvGetReal2D(realPoints, 2, 0), cvGetReal2D(realPoints, 2, 1)), cvPoint(cvGetReal2D(realPoints, 3, 0), cvGetReal2D(realPoints, 3, 1)), CV_RGB(255, 0, 255), 3, 8, 0);
			cvDrawLine(image, cvPoint(cvGetReal2D(realPoints, 3, 0), cvGetReal2D(realPoints, 3, 1)), cvPoint(cvGetReal2D(realPoints, 0, 0), cvGetReal2D(realPoints, 0, 1)), CV_RGB(255, 0, 255), 3, 8, 0);

			cvDrawLine(image, cvPoint(cvGetReal2D(realPoints, 4, 0), cvGetReal2D(realPoints, 4, 1)), cvPoint(cvGetReal2D(realPoints, 5, 0), cvGetReal2D(realPoints, 5, 1)), CV_RGB(255, 0, 255), 3, 8, 0);
			cvDrawLine(image, cvPoint(cvGetReal2D(realPoints, 5, 0), cvGetReal2D(realPoints, 5, 1)), cvPoint(cvGetReal2D(realPoints, 6, 0), cvGetReal2D(realPoints, 6, 1)), CV_RGB(255, 0, 255), 3, 8, 0);
			cvDrawLine(image, cvPoint(cvGetReal2D(realPoints, 6, 0), cvGetReal2D(realPoints, 6, 1)), cvPoint(cvGetReal2D(realPoints, 7, 0), cvGetReal2D(realPoints, 7, 1)), CV_RGB(255, 0, 255), 3, 8, 0);
			cvDrawLine(image, cvPoint(cvGetReal2D(realPoints, 7, 0), cvGetReal2D(realPoints, 7, 1)), cvPoint(cvGetReal2D(realPoints, 4, 0), cvGetReal2D(realPoints, 4, 1)), CV_RGB(255, 0, 255), 3, 8, 0);

			cvDrawLine(image, cvPoint(cvGetReal2D(realPoints, 0, 0), cvGetReal2D(realPoints, 0, 1)), cvPoint(cvGetReal2D(realPoints, 4, 0), cvGetReal2D(realPoints, 4, 1)), CV_RGB(255, 0, 255), 3, 8, 0);
			cvDrawLine(image, cvPoint(cvGetReal2D(realPoints, 1, 0), cvGetReal2D(realPoints, 1, 1)), cvPoint(cvGetReal2D(realPoints, 5, 0), cvGetReal2D(realPoints, 5, 1)), CV_RGB(255, 0, 255), 3, 8, 0);
			cvDrawLine(image, cvPoint(cvGetReal2D(realPoints, 2, 0), cvGetReal2D(realPoints, 2, 1)), cvPoint(cvGetReal2D(realPoints, 6, 0), cvGetReal2D(realPoints, 6, 1)), CV_RGB(255, 0, 255), 3, 8, 0);
			cvDrawLine(image, cvPoint(cvGetReal2D(realPoints, 3, 0), cvGetReal2D(realPoints, 3, 1)), cvPoint(cvGetReal2D(realPoints, 7, 0), cvGetReal2D(realPoints, 7, 1)), CV_RGB(255, 0, 255), 3, 8, 0);
			
		}



		
		//draw windows
		cvShowImage("RGB", image);
		cvShowImage("Binary", imgBW);
			
		}
		freenect_sync_set_led(LED_RED, 0);
		freenect_sync_stop();
		cvFree(&image);
		cvFree(&imgHSV);
		cvFree(&imgThreshed);
		cvFree(&imgBW);
		cvFree(&imgCont);

		terminate_segmenter(&segmenter);
		terminate_treeidmap(&treeidmap);
		terminate_fidtrackerX(&fidtrackerx);

		return EXIT_SUCCESS;
  }