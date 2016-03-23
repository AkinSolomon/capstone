/*@author gihan tharanga*/

#include <iostream>
#include <string>

#include "opencv2\core\core.hpp"
#include "opencv2\contrib\contrib.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2\opencv.hpp"

/*header files*/
#include "FaceRec.h"
//#include "VideoCap.h"
#include "imgProc.h"

using namespace std;
using namespace cv;

int main()
{
	/*come by default, not sure yet*/
	//int x = smoothingImage();

	/*start training*/
	fisherFaceTrainer();

	/*start recognizing*/
	//int value = FaceRecognition();
	system("pause");
	return 0;
}