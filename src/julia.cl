#define MAX_ITER 1000

__kernel void julia(__global uchar4* output,
                         const int width,
                         const int height,
                         const float cRe,
                         const float cIm) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height) {
        return;
    }

    const float remin = -2.0;
    const float remax = 2.0;
    const float immin = -2.0;
    const float immax = 2.0;

    float re = remin + x * (remax - remin)/(width - 1);
    float im = immin + y * (immax - immin)/(height - 1);

    int iter;
    for (iter = 0; iter < MAX_ITER; ++iter) {
        float re2 = re * re;
        float im2 = im * im;

        if (re2 + im2 > 4.0)
            break;

        float newRe = re2 - im2 + cRe;
        float newIm = 2.0 * re * im + cIm;
        re = newRe;
        im = newIm;
    }
    
    uchar4 color;
    if (iter == MAX_ITER) {
        color = (uchar4)(0, 0, 0, 255);
    } else {
        color = (uchar4)(
            (uchar)(255.0f * iter / MAX_ITER),
            (uchar)(255.0f * sqrt((float)iter / MAX_ITER)), 
            (uchar)(255.0f * pow((float)iter / MAX_ITER, 0.3f)), 255);
    }

    output[y * width + x] = color;
}