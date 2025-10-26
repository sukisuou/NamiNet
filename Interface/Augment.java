// Augment.java ~ handles image tweaks for MNIST (by lumi <3)

import java.util.Random;

public class Augment {
    private static final int WIDTH = 28;
    private static final int HEIGHT = 28;
    private static final Random rand = new Random();

    // Shift the image by dx (horizontal) and dy (vertical)
    public static double[] shift(double[] original, int dx, int dy) {
        double[] shifted = new double[WIDTH * HEIGHT];
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                int newX = x + dx;
                int newY = y + dy;
                if (newX >= 0 && newX < WIDTH && newY >= 0 && newY < HEIGHT) {
                    shifted[newY * WIDTH + newX] = original[y * WIDTH + x];
                }
            }
        }
        return shifted;
    }

    // Scale the image by a factor (e.g., 0.9 for zoom out, 1.1 for zoom in)
    public static double[] scale(double[] original, double scaleFactor) {
        double[] scaled = new double[WIDTH * HEIGHT];

        int newW = (int)(WIDTH * scaleFactor);
        int newH = (int)(HEIGHT * scaleFactor);

        int xOffset = (WIDTH - newW) / 2;
        int yOffset = (HEIGHT - newH) / 2;

        for (int y = 0; y < newH; y++) {
            for (int x = 0; x < newW; x++) {
                int srcX = (int)(x / scaleFactor);
                int srcY = (int)(y / scaleFactor);
                if (srcX >= 0 && srcX < WIDTH && srcY >= 0 && srcY < HEIGHT) {
                    int destX = x + xOffset;
                    int destY = y + yOffset;
                    if (destX >= 0 && destX < WIDTH && destY >= 0 && destY < HEIGHT) {
                        scaled[destY * WIDTH + destX] = original[srcY * WIDTH + srcX];
                    }
                }
            }
        }

        return scaled;
    }

    // Rotate the image by a small angle in degrees (-15 to +15 typical)
    public static double[] rotate(double[] original, double angleDegrees) {
        double radians = Math.toRadians(angleDegrees);
        double cos = Math.cos(radians);
        double sin = Math.sin(radians);

        double[] rotated = new double[WIDTH * HEIGHT];

        int cx = WIDTH / 2;
        int cy = HEIGHT / 2;

        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                // Translate point back to origin:
                int tx = x - cx;
                int ty = y - cy;

                // Rotate point
                int rx = (int)Math.round(cos * tx - sin * ty);
                int ry = (int)Math.round(sin * tx + cos * ty);

                // Translate point back
                int newX = rx + cx;
                int newY = ry + cy;

                if (newX >= 0 && newX < WIDTH && newY >= 0 && newY < HEIGHT) {
                    rotated[y * WIDTH + x] = original[newY * WIDTH + newX];
                } else {
                    rotated[y * WIDTH + x] = 0.0; // background
                }
            }
        }

        return rotated;
    }

    // Add Gaussian noise to image
    public static double[] addNoise(double[] original, double stdDev) {
        double[] noisy = new double[original.length];
        for (int i = 0; i < original.length; i++) {
            noisy[i] = original[i] + rand.nextGaussian() * stdDev;
            noisy[i] = Math.max(0.0, Math.min(1.0, noisy[i])); // clamp to [0,1]
        }
        return noisy;
    }

    // Invert image (optional challenge mode)
    public static double[] invert(double[] original) {
        double[] inverted = new double[original.length];
        for (int i = 0; i < original.length; i++) {
            inverted[i] = 1.0 - original[i];
        }
        return inverted;
    }

    // Tiny uniform noise per pixel (e.g., [-scale, +scale])
    public static double[] jitter(double[] original, double scale) {
        double[] jittered = new double[original.length];
        for (int i = 0; i < original.length; i++) {
            jittered[i] = original[i] + (rand.nextDouble() * 2 - 1) * scale;
            jittered[i] = Math.max(0.0, Math.min(1.0, jittered[i])); // clamp
        }
        return jittered;
    }

    //randomly erase a square portion, makes the model less reliant on any one part of the digit
    public static double[] occlude(double[] original, int boxSize) {
        double[] occluded = original.clone();
        int x = rand.nextInt(WIDTH - boxSize);
        int y = rand.nextInt(HEIGHT - boxSize);

        for (int dy = 0; dy < boxSize; dy++) {
            for (int dx = 0; dx < boxSize; dx++) {
                occluded[(y + dy) * WIDTH + (x + dx)] = 0.0;
            }
        }
        return occluded;
    }

    // Elastic distortion for MNIST images
    public static double[] elasticDistort(double[] original, double alpha, double sigma) {
        int width = 28;
        int height = 28;
        Random rand = new Random();

        // Generate random displacement fields
        double[][] dx = new double[height][width];
        double[][] dy = new double[height][width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                dx[y][x] = (rand.nextDouble() * 2 - 1);
                dy[y][x] = (rand.nextDouble() * 2 - 1);
            }
        }

        // Smooth the displacement fields using a simple 3x3 Gaussian filter
        double[][] sdx = gaussianSmooth(dx, sigma);
        double[][] sdy = gaussianSmooth(dy, sigma);

        // Scale by alpha
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                sdx[y][x] *= alpha;
                sdy[y][x] *= alpha;
            }
        }

        // Apply displacement
        double[] distorted = new double[width * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double newX = x + sdx[y][x];
                double newY = y + sdy[y][x];
                distorted[y * width + x] = bilinearInterpolate(original, newX, newY);
            }
        }
        return distorted;
    }

    // Bilinear interpolation for subpixel sampling
    private static double bilinearInterpolate(double[] image, double x, double y) {
        int width = 28;
        int height = 28;

        int x0 = (int)Math.floor(x);
        int y0 = (int)Math.floor(y);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        double wx = x - x0;
        double wy = y - y0;

        double v00 = getPixel(image, x0, y0);
        double v10 = getPixel(image, x1, y0);
        double v01 = getPixel(image, x0, y1);
        double v11 = getPixel(image, x1, y1);

        return (1 - wx)*(1 - wy)*v00 +
            wx*(1 - wy)*v10 +
            (1 - wx)*wy*v01 +
            wx*wy*v11;
    }

    // Get pixel safely (returns 0 if out of bounds)
    private static double getPixel(double[] image, int x, int y) {
        int width = 28;
        int height = 28;
        if (x < 0 || x >= width || y < 0 || y >= height) {
            return 0.0;
        }
        return image[y * width + x];
    }

    // Simple 3x3 Gaussian smoothing
    private static double[][] gaussianSmooth(double[][] field, double sigma) {
        int width = field[0].length;
        int height = field.length;
        double[][] smoothed = new double[height][width];
        double[][] kernel = {
            {1, 2, 1},
            {2, 4, 2},
            {1, 2, 1}
        };
        double norm = 16.0;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double sum = 0.0;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int ny = y + dy;
                        int nx = x + dx;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            sum += field[ny][nx] * kernel[dy + 1][dx + 1];
                        }
                    }
                }
                smoothed[y][x] = sum / norm;
            }
        }
        return smoothed;
    }

    // Apply random augmentation (can be toggled during training)
    public static double[] applyRandom(double[] image) {

        // Randomly shift the image horizontally and vertically by -1, 0, or +1 pixels
        if (rand.nextDouble() < 0.7) {
            int dx = rand.nextInt(3) - 1; // -1,0,1
            int dy = rand.nextInt(3) - 1;
            image = shift(image, dx, dy);
        }

        // Randomly rotate the image by -15 to +15 degrees
        if (rand.nextDouble() < 0.4) {
            double angle = (rand.nextDouble() * 30.0) - 15.0; // -15 to +15 degrees
            image = rotate(image, angle);
        }

        // Randomly add tiny uniform noise to each pixel (jitter)
        if (rand.nextDouble() < 0.2) {  
            image = jitter(image, 0.02);
        }

        // Optionally add Gaussian noise (random small variations) to each pixel
        if (rand.nextDouble() < 0.05) {
            image = addNoise(image, 0.01);
        }

        // With 10% probability, erase a random 3x3 patch (occlusion augmentation)
        if (rand.nextDouble() < 0.10) {
            image = occlude(image, 3);
        }

        // Random scaling (zoom in or out by ~10%)
        if (rand.nextDouble() < 0.3) {
            double scaleFactor = 0.9 + rand.nextDouble() * 0.2;  // range: [0.9, 1.1]
            image = scale(image, scaleFactor);
        }

        // Elastic distortion with 10% chance (MNIST classic)
        if (rand.nextDouble() < 0.2) {
            image = elasticDistort(image, 1.5, 1.0);
        }

        return image;
    }

    // Simple 3x3 smoothing filter for 28x28 image
    public static double[] smooth(double[] original) {
        double[] smoothed = new double[original.length];
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                double sum = 0.0;
                int count = 0;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx >= 0 && nx < WIDTH && ny >= 0 && ny < HEIGHT) {
                            sum += original[ny * WIDTH + nx];
                            count++;
                        }
                    }
                }
                smoothed[y * WIDTH + x] = sum / count;
            }
        }
        return smoothed;
    }
}