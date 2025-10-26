//container for data samples
public class DataSample {
    public double[] inputs;
    public double[] labels;

    public DataSample(double[] inputs, double[] labels) {
        this.inputs = inputs;
        this.labels = labels;
    }
}