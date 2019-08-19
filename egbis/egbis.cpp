/*
Copyright (C) 2015 Yasutomo Kawanishi
Copyright (C) 2013 Christoffer Holmstedt
Copyright (C) 2010 Salik Syed
Copyright (C) 2006 Pedro Felzenszwalb

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/
#include "egbis.h"

#include "egbis/segment-image.h"

#include <cstdio>
#include <fstream>

image<rgb>* convertArrayToNativeImage(float* input, int h, int w){
    //int w = input.cols;
    //int h = input.rows;
    image<rgb> *im = new image<rgb>(w,h);
	memcpy(im->data, input, h * w * sizeof(rgb));
	//~ printf("%d %d\n", h, w);
	//~ for(int i=0; i<h; i++)
    //~ {
        //~ for(int j=0; j<w; j++)
        //~ {
            //~ rgb curr = im->data[i*w+j];
			//~ printf("%d %d %d  ", curr.r, curr.g, curr.b);
        //~ }
        //~ printf("\n");
    //~ }
    return im;
}

bool convertNativeToArray(image<rgb>* input, float* output){
	int w = input->width();
	int h = input->height();
    memcpy(output, input->data, h * w * sizeof(rgb));
    
}

extern "C" {
PyObject* runEgbisOnMat(float* input, int h, int w, float* output, float sigma, float k, int min_size, int &numccs) {
	printf("h=%d w=%d sigma=%f k=%f min_size=%d\n", h, w, sigma, k, min_size);
    // 1. Convert to native format
    printf("1. Convert to native format\n");
    image<rgb> *nativeImage = convertArrayToNativeImage(input, h, w);
    // 2. Run egbis algoritm
    printf("2. Run egbis algorithm\n");
    int n = 0;
    std::vector<std::vector<std::pair<int,int>>> segs;
    image<rgb> *segmentedImage = segment_image(nativeImage, sigma, k, min_size, &n, segs);
    numccs = n;
    // 3. Convert back to Mat format
    printf("3. Convert back to Mat format\n");
    convertNativeToArray(segmentedImage, output);

	delete nativeImage;
	delete segmentedImage;

    std::sort(segs.begin(), segs.end(),[](const auto& s1, const auto& s2 ){
        return s1.size() < s2.size();
    });
    std::reverse(segs.begin(), segs.end());

    PyObject* x = PyList_New(0);
    PyObject* y = PyList_New(0);
    PyObject* sizes = PyList_New(0);

    int i=0;
    printf("size: %ud \n", segs.size());
    for (auto& vec : segs) {

        if (vec.empty())
            continue;
        //PyObject *subList = PyTuple_New(vec.size());
        int c =0;
        PyList_Append(sizes, PyLong_FromLong(vec.size()));
        for (auto& pair : vec) {
            //printf("%d %d\n", set.first, set.second);
            PyList_Append(x, PyLong_FromLong(pair.first));
            PyList_Append(y, PyLong_FromLong(pair.second));
            //PyTuple_SetItem(subList, c++, Py_BuildValue("(i i)", c, c));
        }
    }

    return Py_BuildValue("(N (N N))", sizes, x, y);


}
PyObject* foo()
{
    PyObject* result = PyList_New(0);
    int i;


    for (i = 0; i < 100000; ++i)
    {
//        PyObject* subList = PyList_New(0);
        for (int j=0; j < 5; j++)
            PyList_Append(result, PyLong_FromLong(i*5 + j));
//        PyList_Append(result, PyTuple_Pack(2, Py_BuildValue("i", i), Py_BuildValue("i", j)));
//            PyList_Append(result, Py_BuildValue("(i i)", i , j));
//        PyList_Append(result, subList);
//        delete subList;
//        PyList_Append(result, PyLong_FromLong(i));
    }

    return result;
}
}

int main() {
	std::ifstream cin("../temp.txt");
	int h,w;
	
	cin >> h >> w;
	printf("%d %d\n", h, w);
	image<rgb> *im = new image<rgb>(w,h);
    for(int i=0; i<h; i++)
    {
        for(int j=0; j<w; j++)
        {
            rgb curr;
            cin>>curr.r>>curr.g>>curr.b;
            im->data[i*w+j] = curr;
        }
    }
    int numccs;
    std::vector<std::vector<std::pair<int,int>>> segs;
    image<rgb> *segmentedImage = segment_image(im, 0.30, 200 , 200, &numccs, segs);
    std::vector<int> vec{1,2,3,4,5,6,7,8,9, 10};
    //foo();


	return 0;
}
