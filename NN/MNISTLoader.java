import java.io.*;
import java.util.*;

public class MNISTLoader{
    
    // A simple data holder
    public static class Sample{
        public double[] input;  // 784 pixels
        public int label;       // 0-9
        public Sample(double[] input, int label){
            this.input = input;
            this.label = label;
        }
    }

    public static List<Sample> loadMNISTCSV(String filePath) throws IOException{
        List<Sample> dataset = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new FileReader(filePath));

        String line;
        boolean skipHeader = true;

        while ((line = reader.readLine()) != null) {
            // Skip header row
            if (skipHeader) {
                skipHeader = false;
                continue;
            }

            String[] parts = line.split(",");

            int label = Integer.parseInt(parts[0]);

            double[] input = new double[784];
            for (int i = 0; i < 784; i++) {
                int pixel = Integer.parseInt(parts[i + 1]);
                input[i] = pixel / 255.0; // normalize to 0-1
            }

            dataset.add(new Sample(input, label));
        }
        reader.close();

        return dataset;
    }
}
