//math functions        ~template~

class Functions{
    //activation functions
    public static double ReLU(double a){
        return a >= 0 ? a : 0;
    }
    public static double ReLUDerivatives(double a){
        return a >= 0 ? 1 : 0;
    }

    public static double leakyReLU(double a){
        return a >= 0 ? a : 0.05*a;
    }
    public static double leakyReLUDerivatives(double a){
        return a >= 0 ? 1 : 0.05;
    }

    public static double sigmoid(double a){
        return 1 / (1 + Math.exp(-a));
    }
    public static double sigmoidDerivative(double a){
        double s = sigmoid(a);
        return s * (1 - s);
    }

    public static double tanh(double a){
        return Math.tanh(a);
    }
    public static double tanhDerivative(double a){
        double t = Math.tanh(a);
        return 1 - t * t;
    }

    //softmax - output
    public static double[] softmax(double[] input){
        double max = Double.NEGATIVE_INFINITY;

        //find max in input
        for(double i : input){
            if (i > max) max = i;
        }

        double sum = 0.0;
        double[] exps = new double[input.length];
        //calculate exponentials and sum of it
        for(int i=0; i<input.length; i++){
            exps[i] = Math.exp(input[i] - max);
            sum += exps[i];
        }

        //normalize
        for(int i=0; i<input.length; i++){
            exps[i] /= sum;
        }

        return exps;
    }

    //MSE
    public static double meanSquaredError(double[] yp, double[] yt){
        double loss = 0.0;
        for(int i=0; i<yp.length; i++){
            double diff = yp[i] - yt[i];
            loss += diff * diff;
        }
        return loss / yp.length;
    }
    public static double[] meanSquaredErrorDerivatives(double[] yp, double[] yt){
        double[] grad = new double[yp.length];
        for(int i=0; i<yp.length; i++){
            grad[i] = 2 * (yp[i] - yt[i]) / yp.length;
        }
        return grad;
    }

    //cross-entropy (used with softmax)
    public static double crossEntropyLoss(double[] yp, double[] yt){
        double epsilon = 1e-8;      //avoid log(0)
        double loss = 0.0;
        for(int i=0; i<yp.length; i++){
            double p = Math.max(epsilon, yp[i]);
            double q = Math.max(epsilon, yt[i]);
            loss += q * Math.log(p);
        }

        return -loss;
    }
    public static double[] crossEntropyLossDerivatives(double[] yp, double[] yt){
        double[] grad = new double[yp.length];
        for(int i=0; i<yp.length; i++){
            grad[i] = yp[i] - yt[i];
        }
        return grad;
    }
}