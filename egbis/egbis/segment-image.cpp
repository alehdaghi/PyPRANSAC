/*
Copyright (C) 2015 Yasutomo Kawanishi
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

#include <map>
#include "segment-image.h"

// random color
rgb random_rgb(){
    rgb c;
    double r;

    c.r = (uchar)random();
    c.g = (uchar)random();
    c.b = (uchar)random();

    return c;
}

// dissimilarity measure between pixels
static inline float diff(image<float> *r, image<float> *g, image<float> *b,
                         int x1, int y1, int x2, int y2) {
    return sqrt(square(imRef(r, x1, y1)-imRef(r, x2, y2)) +
                square(imRef(g, x1, y1)-imRef(g, x2, y2)) +
                square(imRef(b, x1, y1)-imRef(b, x2, y2)));
}

/*
 * Segment an image
 *
 * Returns a color image representing the segmentation.
 *
 * im: image to segment.
 * sigma: to smooth the image.
 * c: constant for treshold function.
 * min_size: minimum component size (enforced by post-processing stage).
 * num_ccs: number of connected components in the segmentation.
 */
universe *segmentation(image<rgb> *im, float sigma, float c, int min_size,
                       int *num_ccs) {
    int width = im->width();
    int height = im->height();

    image<float> *r = new image<float>(width, height);
    image<float> *g = new image<float>(width, height);
    image<float> *b = new image<float>(width, height);

    // smooth each color channel
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            imRef(r, x, y) = imRef(im, x, y).r;
            imRef(g, x, y) = imRef(im, x, y).g;
            imRef(b, x, y) = imRef(im, x, y).b;
        }
    }
    image<float> *smooth_r = smooth(r, sigma);
    image<float> *smooth_g = smooth(g, sigma);
    image<float> *smooth_b = smooth(b, sigma);
    delete r;
    delete g;
    delete b;

    // build graph
    std::vector<edge> edges;
    edges.reserve(width*height*4);
    //edge *edges = new edge[width*height*4];
    int numEdges = 0;
    int numVerts = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (std::isnan(imRef(smooth_r, x, y)))
                continue;
            numVerts++;
            if (x < width-1) {
                if (std::isnan(imRef(smooth_r, x + 1, y)))
                    continue;
                edges.push_back({y * width + x, y * width + (x + 1), diff(smooth_r, smooth_g, smooth_b, x, y, x + 1, y)});
                numEdges++;
            }

            if (y < height-1) {
                if (std::isnan(imRef(smooth_r, x , y + 1)))
                    continue;
                edges.push_back({y * width + x, (y + 1) * width + x , diff(smooth_r, smooth_g, smooth_b, x, y, x , y + 1)});
                numEdges++;
            }

            if ((x < width-1) && (y < height-1)) {
                if (std::isnan(imRef(smooth_r, x + 1, y + 1)))
                    continue;
                edges.push_back({y * width + x, (y + 1) * width + (x + 1), diff(smooth_r, smooth_g, smooth_b, x, y, x + 1, y + 1)});
                numEdges++;
            }

            if ((x < width-1) && (y > 0)) {
                if (std::isnan(imRef(smooth_r, x + 1, y - 1)))
                    continue;
                edges.push_back({y * width + x, (y - 1) * width + (x + 1), diff(smooth_r, smooth_g, smooth_b, x, y, x + 1, y - 1)});
                numEdges++;
            }
        }
    }
    delete smooth_r;
    delete smooth_g;
    delete smooth_b;

    // segment
    universe *u = segment_graph(width*height, numEdges, edges, c);

    // post process small components
    for (int i = 0; i < numEdges; i++) {
        int a = u->find(edges[i].a);
        int b = u->find(edges[i].b);
        if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
            u->join(a, b);
    }
    //delete [] edges;
    *num_ccs = u->num_sets();

    return u;
}

image<rgb>* visualize(universe *u, int width, int height, std::vector<std::vector<std::pair<int,int>>>& segs, image<rgb> *input){
    image<rgb> *output = new image<rgb>(width, height);

    // pick random colors for each component
    rgb *colors = new rgb[width*height];
    //std::map<int, int> ids;
    int * ids = new int[width*height];
    memset(ids, 0, sizeof(int) * width*height);
    int cc = 1;
    for (int i = 0; i < width*height; i++)
        colors[i] = random_rgb();

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (std::isnan(imRef(input, x, y).r) ) {
                imRef(output, x, y) = {0, 0, 0};
                continue;
            }
            int comp = u->find(y * width + x);

            imRef(output, x, y) = colors[comp];
            if (ids[comp] == 0) {
                ids[comp] = cc++;
            }
            segs[ids[comp] - 1].push_back(std::make_pair(x,y));
        }
    }

    delete [] colors;
    return output;
}

image<rgb> *segment_image(image<rgb> *im, float sigma, float c, int min_size,
                          int *num_ccs, std::vector<std::vector<std::pair<int,int>>>& segs) {
    universe *u = segmentation(im, sigma, c, min_size, num_ccs);
    segs.resize(*num_ccs);
    image<rgb> *visualized = visualize(u, im->width(), im->height(), segs, im);
    delete u;
    return visualized;
}
