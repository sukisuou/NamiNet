//runs the training process (caution)

import java.io.*;
import java.util.*;

public class Trainer{
    public static void main(String[] args) throws IOException{
        //warnings
        Scanner in = new Scanner(System.in);
        System.out.println("\nWarning : This is a training code, do NOT run it unauthorized.");
        System.out.print("You sure you wanna run this? (y/n) : ");
        String perm = in.nextLine();

        if(perm.trim().equalsIgnoreCase("y")){
            System.out.print("Are you REALLY sure? (y/n) : ");
            String perm2 = in.nextLine();

            if(!perm2.trim().equalsIgnoreCase("y")){
                System.out.println("Phew, dodged a bullet.");
                return;
            }
        }else{
            System.out.println("Phew, dodged a bullet.");
            return;
        }
        
        //load data from dataset
        List<MNISTLoader.Sample> rawSamples = MNISTLoader.loadMNISTCSV("mnist_train.csv");
        List<DataSample> dataset = new ArrayList<>();

        for (MNISTLoader.Sample sample : rawSamples) {
            // Convert label to one-hot
            double[] labelOneHot = new double[10];
            labelOneHot[sample.label] = 1.0;

            dataset.add(new DataSample(sample.input, labelOneHot));
        }

        //testing value
        System.out.print("Testing? (y/n): ");
        String testComf = in.nextLine();
        boolean test;
        if(testComf.trim().equalsIgnoreCase("y")){
            System.out.print("Sample size (1-60000): ");
            int sampleSize = in.nextInt();
            in.nextLine();  //clear buffer because java is whiny like that
            if (dataset.size() >= sampleSize && sampleSize > 0) {
                dataset = dataset.subList(0, sampleSize);
            }else{
                System.out.println("You done messed up, boy.");
            }
        }System.out.println("Loaded: " + dataset.size() + " samples.\n");
        
        //network architechture                                                     ~ big part
        double[] dropoutRates = new double[]{0.1, 0.05, 0.0};    //rates of neuron dropout per layer
        NeuralNetwork naminet = new NeuralNetwork(new int[]{784, 128, 64, 10}, dropoutRates);

        //training parameters
        int epochs = 100;
        double initialLearningRate = 0.002;
        double decayRate = 0.998;

        Random rand = new Random();
        long startTime = System.currentTimeMillis();
        double[] avgLossAll = new double[epochs];
        double[] accuracyAll = new double[epochs];

        //training loop
        System.out.println("~ Training Session ~");
        for(int epoch=1; epoch<=epochs; epoch++){
            Collections.shuffle(dataset, rand); //shuffle for randomness
            double totalLoss = 0.0;
            int correct = 0;
            double learningRate = initialLearningRate * Math.pow(decayRate, epoch);
            learningRate = Math.max(0.0005, learningRate);

            for(DataSample sample : dataset){
                //augment the images first
                double[] augmentedInput = Augment.applyRandom(sample.inputs);
                double[] smoothedInput = Augment.smooth(augmentedInput);

                //forward pass
                double[] prediction = naminet.forward(smoothedInput);

                //compute loss
                double loss = Functions.crossEntropyLoss(prediction, sample.labels);
                totalLoss += loss;

                //compute accuracy
                int predictedClass = argMax(prediction);
                int trueClass = argMax(sample.labels);
                if(predictedClass == trueClass){
                    correct++;
                }

                //backpropagation
                naminet.train(smoothedInput, sample.labels, learningRate);
            }

            //report average loss and accuracy
            double avgLoss = totalLoss / dataset.size();
            avgLossAll[epoch-1] = avgLoss;
            double accuracy = 100.0 * correct / dataset.size();
            accuracyAll[epoch-1] = accuracy;
            double diff = (epoch != 1) ? avgLossAll[epoch-2] - avgLossAll[epoch-1] : 0;   //difference in loss
            String sign = (diff >= 0) ? "+" : "-";
            double diffPercent = 100.0 * diff;
            System.out.printf("Epoch %03d - Avg loss: %.6f (%s%.2f%%) - Accuracy: %.2f%%\n", 
                                epoch, avgLoss, sign, Math.abs(diffPercent), accuracy);

            //giving my lil cpu some nappy time
            int mimir = 300000;
            int boop = 30000;
            try{
                if(epoch != epochs){
                    if (epoch % 10 == 0) Thread.sleep(mimir);    //sleep for 5 minutes per 10 epochs
                    else Thread.sleep(boop);   //sleep for 30 seconds per epoch
                }
            }catch(InterruptedException e){
                e.printStackTrace();
            }
        }

        long endTime = System.currentTimeMillis();
        double seconds = (endTime - startTime) / 1000.0;
        System.out.printf("\nTraining complete in %.2f seconds. (%d samples)\n", seconds, dataset.size());
        
        //saving the model
        ModelSaver.saveModel(naminet, "naminet_model.bin");

        //clear the logging file
        try (PrintWriter pw = new PrintWriter("naminet_training_log.txt")){
            pw.print("");
        } catch (IOException e) {
            e.printStackTrace();
        }

        //saving training log
        try(FileWriter fw = new FileWriter("naminet_training_log.txt", true);
            BufferedWriter bw = new BufferedWriter(fw);
            PrintWriter out = new PrintWriter(bw)){
                out.println("Training log:");
                
                //loss
                out.print("\n~ Loss");
                for(int i=0; i<epochs; i++){
                    if(i%10 == 0){
                        out.printf("%n");
                    }out.printf("%.6f", avgLossAll[i]);
                    if(i != epochs-1) out.print(", ");
                }out.printf("%n");

                //accuracy
                out.print("\n~ Accuracy (%)");
                for(int i=0; i<epochs; i++){
                    if(i%10 == 0){
                        out.printf("%n");
                    }out.printf("%.2f", accuracyAll[i]);
                    if(i != epochs-1) out.print(", ");
                }out.printf("%n");
        }catch(IOException e){
            e.printStackTrace();
        }

        //time keeper
        try(FileWriter fw = new FileWriter("naminet_training_log.txt", true);
                BufferedWriter bw = new BufferedWriter(fw);
                PrintWriter out = new PrintWriter(bw)){
                    out.printf("%n%n- Finished in %.02f seconds. (%d samples)", seconds, dataset.size());
                    System.out.println("\n(Training session logged into naminet_training_log.txt)");
        }catch(IOException e){
            e.printStackTrace();
        }
    }

    public static int argMax(double[] array){   //finding biggest value in an array (argument)
        int index = 0;
        for(int i=1; i<array.length; i++){
            if(array[i] > array[index]){
                index = i;
            }
        }
        return index;
    }
}