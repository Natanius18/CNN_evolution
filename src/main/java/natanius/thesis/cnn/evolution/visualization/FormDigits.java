package natanius.thesis.cnn.evolution.visualization;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.image.BufferedImage;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.WindowConstants;
import natanius.thesis.cnn.evolution.network.NeuralNetwork;

public class FormDigits extends JFrame implements Runnable, MouseListener, MouseMotionListener, KeyListener {

    private static final int W = 28;
    private static final int H = 28;
    private static final int SCALE = 32;

    private int mousePressed = 0;
    private int mx = 0;
    private int my = 0;
    private double[][] colors = new double[W][H];

    private final transient BufferedImage img = new BufferedImage(W * SCALE + 200, H * SCALE, BufferedImage.TYPE_INT_RGB);
    private final BufferedImage pimg = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB);
    private final NeuralNetwork nn;

    public FormDigits(NeuralNetwork nn) {
        this.nn = nn;
        this.setSize(W * SCALE + 200 + 16, H * SCALE + 38);
        this.setVisible(true);
        this.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        this.setLocation(50, 50);
        this.add(new JLabel(new ImageIcon(img)));
        addMouseListener(this);
        addMouseMotionListener(this);
        addKeyListener(this);
    }

    @Override
    public void run() {
        while (true) {
            this.repaint();
        }
    }

    @Override
    public void paint(Graphics g) {
        double[] inputs = new double[784];
        for (int i = 0; i < W; i++) {
            for (int j = 0; j < H; j++) {
                if (mousePressed != 0) {
                    double dist = (i - mx) * (i - mx) + (j - my) * (j - my);
                    if (dist < 1) dist = 1;
                    dist *= dist;
                    if (mousePressed == 1) colors[i][j] += 0.1 / dist;
                    else colors[i][j] -= 0.1 / dist;
                    if (colors[i][j] > 1) colors[i][j] = 1;
                    if (colors[i][j] < 0) colors[i][j] = 0;
                }
                int color = (int) (colors[i][j] * 255);
                color = (color << 16) | (color << 8) | color;
                pimg.setRGB(i, j, color);
                inputs[i + j * W] = colors[i][j];
            }
        }
        double[] outputs = nn.guessInRealTime(inputs);
        int maxDigit = 0;
        double maxDigitWeight = -1;
        for (int i = 0; i < 10; i++) {
            if (outputs[i] > maxDigitWeight) {
                maxDigitWeight = outputs[i];
                maxDigit = i;
            }
        }
        Graphics2D ig = (Graphics2D) img.getGraphics();
        ig.drawImage(pimg, 0, 0, W * SCALE, H * SCALE, this);
        ig.setColor(Color.lightGray);
        ig.fillRect(W * SCALE + 1, 0, 200, H * SCALE);
        ig.setFont(new Font("TimesRoman", Font.BOLD, 48));
        for (int i = 0; i < 10; i++) {
            ig.setColor(maxDigit == i ? Color.RED : Color.GRAY);
            ig.drawString(i + ":", W * SCALE + 20, i * W * SCALE / 15 + 150);
            Color rectColor = new Color(0, (float) outputs[i], 0);
            int rectWidth = (int) (outputs[i] * 100);
            ig.setColor(rectColor);
            ig.fillRect(W * SCALE + 70, i * W * SCALE / 15 + 122, rectWidth, 30);
        }
        g.drawImage(img, 8, 30, W * SCALE + 200, H * SCALE, this);
    }

    @Override
    public void mouseClicked(MouseEvent e) {

    }

    @Override
    public void mousePressed(MouseEvent e) {
        mousePressed = 1;
        if (e.getButton() == 3) mousePressed = 2;
    }

    @Override
    public void mouseReleased(MouseEvent e) {
        mousePressed = 0;
    }

    @Override
    public void mouseEntered(MouseEvent e) {

    }

    @Override
    public void mouseExited(MouseEvent e) {

    }

    @Override
    public void keyTyped(KeyEvent e) {

    }

    @Override
    public void keyPressed(KeyEvent e) {
        if (e.getKeyCode() == KeyEvent.VK_SPACE) {
            colors = new double[W][H];
        }
    }

    @Override
    public void keyReleased(KeyEvent e) {

    }

    @Override
    public void mouseDragged(MouseEvent e) {
        mx = e.getX() / SCALE;
        my = e.getY() / SCALE;
    }

    @Override
    public void mouseMoved(MouseEvent e) {
        mx = e.getX() / SCALE;
        my = e.getY() / SCALE;
    }
}
