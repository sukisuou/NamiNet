//swing interface for NamiNet

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;

public class NamiNetGUI extends JFrame{
    private DrawPanel drawPanel;
    private JLabel headerLabel;
    private JLabel predictionLabel;
    private JLabel confidenceLabel;
    private NeuralNetwork naminet;

    public NamiNetGUI(NeuralNetwork naminet){
        this.naminet = naminet;

        setUndecorated(true);
        
        //custom title bar (no windows ew)
        JPanel titleBar = new JPanel();
        titleBar.setBackground(new Color(220, 200, 255));
        titleBar.setLayout(new BorderLayout());
        titleBar.setPreferredSize(new Dimension(400, 30));
        setTitle("NamiNet");
        setIconImage(Toolkit.getDefaultToolkit().getImage("naminet_icon.png"));
        getRootPane().setBorder(BorderFactory.createLineBorder(new Color(220, 200, 255), 2, true));
        
        //title name
        Image rawIcon = new ImageIcon("naminet_icon.png").getImage();
        Image scaledIcon = rawIcon.getScaledInstance(30, 30, Image.SCALE_SMOOTH);
        ImageIcon namiIcon = new ImageIcon(scaledIcon);
        JLabel titleLabel = new JLabel("~ Naminet", namiIcon, JLabel.LEFT);
        titleLabel.setFont(new Font("Monospaced", Font.BOLD, 14));
        titleLabel.setBorder(BorderFactory.createEmptyBorder(2, 5, 2, 5));

        //x button
        JButton closeButton = new JButton("X");
        closeButton.setFocusPainted(false);
        closeButton.setBorderPainted(false);
        closeButton.setContentAreaFilled(true);
        closeButton.setOpaque(true);
        closeButton.setMargin(new Insets(0, 0, 0, 0));
        closeButton.setPreferredSize(new Dimension(30, 30));
        closeButton.setBackground(new Color(255, 0, 0));
        closeButton.setForeground(Color.BLACK);
        closeButton.setFont(new Font("Monospaced", Font.BOLD, 14));
        closeButton.setToolTipText("Exit the application.");
        closeButton.addActionListener(e -> System.exit(0));

        //dragging
        final Point dragPoint = new Point();
        titleBar.addMouseListener(new MouseAdapter(){
            public void mousePressed(MouseEvent e){
                dragPoint.x = e.getX();
                dragPoint.y = e.getY();
            }
        });
        titleBar.addMouseMotionListener(new MouseMotionAdapter(){
            public void mouseDragged(MouseEvent e){
                Point p = getLocation();
                setLocation(p.x + e.getX() - dragPoint.x, p.y + e.getY() - dragPoint.y);
            }
        });
        
        //add them together
        titleBar.add(titleLabel, BorderLayout.WEST);
        titleBar.add(closeButton, BorderLayout.EAST);

        //create a drawing panel
        drawPanel = new DrawPanel();
        drawPanel.setPreferredSize(new Dimension(280, 280));

        //label for prediction
        headerLabel = new JLabel("Prediction:");
        headerLabel.setHorizontalAlignment(SwingConstants.CENTER);
        headerLabel.setFont(new Font("Monospaced", Font.BOLD, 16));
        predictionLabel = new JLabel("Draw a digit (0-9)");
        predictionLabel.setHorizontalAlignment(SwingConstants.CENTER);
        predictionLabel.setFont(new Font("Monospaced", Font.BOLD, 16));
        predictionLabel.setForeground(Color.BLACK);
        confidenceLabel = new JLabel("");
        confidenceLabel.setHorizontalAlignment(SwingConstants.CENTER);
        confidenceLabel.setFont(new Font("Monospaced", Font.BOLD, 16));
        confidenceLabel.setForeground(Color.BLACK);

        //wraps the prediction to avoid adjustment
        JPanel predictionWrapper = new JPanel(null);
        predictionWrapper.setPreferredSize(new Dimension(220, 30));
        predictionWrapper.setMaximumSize(new Dimension(220, 30));
        predictionWrapper.setMinimumSize(new Dimension(220, 30));

        //puts the label in the wrapper with setBounds
        predictionLabel.setBounds(0, 0, 220, 30);
        predictionWrapper.add(predictionLabel);
        predictionWrapper.setBackground(new Color(245, 210, 235));

        JPanel confidenceWrapper = new JPanel(null);
        confidenceWrapper.setPreferredSize(new Dimension(220, 30));
        confidenceWrapper.setMaximumSize(new Dimension(220, 30));
        confidenceWrapper.setMinimumSize(new Dimension(220, 30));

        confidenceLabel.setBounds(0, 0, 220, 30);
        confidenceWrapper.add(confidenceLabel);
        confidenceWrapper.setBackground(new Color(245, 210, 235));

        //clear button
        JButton clearButton = new JButton("Clear");
        clearButton.addActionListener(e -> drawPanel.clear());
        clearButton.setToolTipText("Clears the canvas.");
        clearButton.setFocusPainted(false);
        clearButton.setBackground(new Color(255, 200, 200));
        clearButton.setForeground(Color.BLACK);
        clearButton.setFont(new Font("Monospaced", Font.BOLD, 14));

        //predict button
        JButton predictButton = new JButton("Predict");
        predictButton.setToolTipText("Predicts the digit you just drew.");
        predictButton.setFocusPainted(false);
        predictButton.setBackground(new Color(190, 230, 255));
        predictButton.setForeground(Color.BLACK);
        predictButton.setFont(new Font("Monospaced", Font.BOLD, 14));

        //prediction (passed thru the network)
        predictButton.addActionListener(e -> { 
            double[] input = drawPanel.getNormalizedInput();    //get the raw pixel input
            double[] smoothedInput = Augment.smooth(input);     //smoothened the raw input
            double[] output = naminet.predict(smoothedInput);   //predict with naminet

            //gui output
            int prediction = argMax(output);
            String confidence = String.format("%.2f", (output[prediction] * 100.0));
            String article = (prediction == 8) ? "an" : "a";    //pesky grammar police
            if(output[prediction] < 0.3){
                predictionLabel.setText("Sorry, no idea...");
            }else if(output[prediction] < 0.5){
                predictionLabel.setText("Uhh, is it " + article + " " + prediction + "?");
            }else if(output[prediction] < 0.8){
                predictionLabel.setText("Its probably " + article + " " + prediction);
            }else{
                predictionLabel.setText("I think its " + article + " " + prediction + "!");
            }
            confidenceLabel.setText("(" + confidence + "% confidence)");

            //terminal output
            for(int i=0; i<784; i++){
                if(i%28 == 0){
                    System.out.println();
                }
                System.out.print((smoothedInput[i] > 0.5 ? '.' : '#') + " ");
            }System.out.println();

            //probabilities for all 10 digits
            System.out.println("\nProbabilities:");
            for (int i = 0; i < output.length; i++) {
                System.out.printf("%d: %.4f  ", i, output[i]);
            }System.out.println();
        });

        //buttons panel
        JPanel buttonPanel = new JPanel(new BorderLayout());
        JPanel buttonRow = new JPanel();
        buttonRow.add(clearButton);
        buttonRow.add(predictButton);
        buttonPanel.add(buttonRow, BorderLayout.CENTER);
        buttonRow.setBackground(new Color(220, 200, 255));

        //labels panel
        JPanel labelPanel = new JPanel(new BorderLayout());
        JPanel labelRow = new JPanel();
        labelRow.setLayout(new BoxLayout(labelRow, BoxLayout.Y_AXIS));
        labelRow.setPreferredSize(new Dimension(250, 30));
        labelRow.setMaximumSize(new Dimension(250, 30));
        labelRow.setMinimumSize(new Dimension(250, 30));

        headerLabel.setAlignmentX(Component.RIGHT_ALIGNMENT);
        predictionLabel.setAlignmentX(Component.CENTER_ALIGNMENT);
        confidenceLabel.setAlignmentX(Component.CENTER_ALIGNMENT);

        labelRow.add(headerLabel);
        labelRow.add(Box.createRigidArea(new Dimension(0, 5)));
        labelRow.add(predictionWrapper);
        labelRow.add(Box.createRigidArea(new Dimension(0, 5)));
        labelRow.add(confidenceWrapper);
        labelPanel.add(labelRow, BorderLayout.CENTER);
        labelRow.setBackground(new Color(245, 210, 235));
        labelRow.setBorder(BorderFactory.createLineBorder(new Color(245, 210, 235), 2, true));

        //layout : panel in middle, button down
        setLayout(new BorderLayout());
        add(titleBar, BorderLayout.NORTH);
        add(drawPanel, BorderLayout.CENTER);
        add(buttonPanel, BorderLayout.SOUTH);
        add(labelPanel, BorderLayout.EAST);

        pack();
        setLocationRelativeTo(null);
        setVisible(true);
    }

    private class DrawPanel extends JPanel{
        private BufferedImage image;
        private Graphics2D g2d;
        private int res = 56;
        private Point lastPoint = null;

        //draws with mouse input
        public DrawPanel(){
            image = new BufferedImage(res, res, BufferedImage.TYPE_BYTE_GRAY);
            g2d = image.createGraphics();
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2d.setBackground(Color.BLACK);     //color for panel
            g2d.clearRect(0, 0, res, res);
            g2d.setColor(Color.WHITE);          //color for writing

            // Mouse listener for drawing
            addMouseMotionListener(new MouseMotionAdapter(){
                @Override
                public void mouseDragged(MouseEvent mouse){
                    int w = getWidth();
                    int h = getHeight();
                    if (w == 0 || h == 0) {
                        //if the panel hasn't been laid out yet, skip this event
                        return;
                    }

                    int x = mouse.getX() * res / w;
                    int y = mouse.getY() * res / h;
                    Point currentPoint = new Point(x, y);
                    int size1 = 2;
                    int size2 = 4;
                    int size3 = 6;

                    if (lastPoint != null) {
                        int dx = currentPoint.x - lastPoint.x;
                        int dy = currentPoint.y - lastPoint.y;
                        int steps = Math.max(Math.abs(dx), Math.abs(dy));
                        if(steps == 0){
                            g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 0.4f));
                            g2d.fillOval(currentPoint.x - 2, currentPoint.y - 2, size3, size3);

                            g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 0.7f));
                            g2d.fillOval(currentPoint.x - 1, currentPoint.y - 1, size2, size2);

                            g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));
                            g2d.fillOval(currentPoint.x, currentPoint.y, size1, size1);
                        }else{
                            for (int i = 0; i <= steps; i++) {
                                int interpX = lastPoint.x + i * dx / steps;
                                int interpY = lastPoint.y + i * dy / steps;

                                g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 0.4f));
                                g2d.fillOval(interpX - 2, interpY - 2, size3, size3);

                                g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 0.7f));
                                g2d.fillOval(interpX - 1, interpY - 1, size2, size2);

                                g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));
                                g2d.fillOval(interpX, interpY, size1, size1);
                            }
                        }
                    }

                    lastPoint = currentPoint;
                    repaint();
                }
            });

            addMouseListener(new MouseAdapter() {
                @Override
                public void mouseReleased(MouseEvent e) {
                    lastPoint = null; // Reset when stroke ends
                }
            });
        }

        //repaint the panel
        @Override
        protected void paintComponent(Graphics g){
            super.paintComponent(g);

            // Scale the 28x28 image to fit the panel
            g.drawImage(image, 0, 0, getWidth(), getHeight(), null);
        }

        //clears the panel
        public void clear(){
            g2d.clearRect(0, 0, res, res);
            repaint();
        }

        //convert to grayscale values (for neurons to computate)
        public double[] getNormalizedInput(){
            //scales the canvas
            BufferedImage small = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
            Graphics2D g2 = small.createGraphics();
            g2.drawImage(image, 0, 0, 28, 28, null);
            g2.dispose();

            //read input from scaled image
            double[] input = new double[28 * 28];   //28 by 28
            for (int y = 0; y < 28; y++) {
                for (int x = 0; x < 28; x++) {
                    int rgb = small.getRGB(x, y) & 0xFF;
                    input[y * 28 + x] = rgb / 255.0;
                }
            }

            return input;        
        }
    }

    private int argMax(double[] array){
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}