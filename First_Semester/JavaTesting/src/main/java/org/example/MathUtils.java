package org.example;

public class MathUtils {

    // Returns the factorial of a non-negative integer
    public static long factorial(int n) {
        if (n < 0) {
            throw new IllegalArgumentException("n must be non-negative");
        }
        long result = 1;
        for (int i = 2; i <= n; i++) {
            result *= i;
        }
        return result;
    }

    // Returns true if number is prime
    public static boolean isPrime(int n) {
        if (n <= 1) return false;
        if (n == 2) return true;
        if (n % 2 == 0) return false;
        for (int i = 3; i * i <= n; i += 2) {
            if (n % i == 0) return false;
        }
        return true;
    }

    // Returns average of numbers, throws exception if array is null or empty
    public static double average(double... nums) {
        if (nums == null || nums.length == 0) {
            throw new IllegalArgumentException("Array must not be null or empty");
        }
        double sum = 0;
        for (double n : nums) sum += n;
        return sum / nums.length;
    }

    // Returns absolute difference
    public static double absDiff(double a, double b) {
        double diff = a - b;
        return diff < 0 ? -diff : diff;
    }
}
