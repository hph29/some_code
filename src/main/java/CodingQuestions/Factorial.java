package CodingQuestions;

public class Factorial {

    public static int factorialRecursive(int n){
        if (n == 1){
            return 1;
        }
        else{
            return n * factorialRecursive(n-1);
        }
    }

    public static int factorialIterative(int n){
        int r = 1;
        while(n > 1){
            r = r * n;
            n--;
        }
        return r;
    }

    // Given a number N return the index value of the Fibonacci sequence, where the sequence is:

    // 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144 ...
    // the pattern of the sequence is that each value is the sum of the 2 previous values, that means that for N=5 → 2+3

    //For example: fibonacciRecursive(6) should return 8

    public static int fibonacciIterative(int n){
        int i = 1;
        int j = 0;
        if (n == 0){
            return 0;
        }
        else if (n == 1){
            return 1;
        }
        else{
            while(n > 1){
                int tmp = i;
                i = i + j;
                j = tmp;
                n--;
            }
            return i;
        }
        //code here;
    }

    public static int fibonacciRecursive(int n) {
        //code here;
        if (n == 0){
            return 0;
        }
        else if (n == 1){
            return 1;
        }
        else{
            return fibonacciRecursive(n-1) + fibonacciIterative(n - 2);
        }
    }

    public static String reverseStringRecursive(String str){
        return rec(str, "");
    }

    public static String rec(String originalString, String reversedString){
        if (originalString.length() == 0){
            return reversedString;
        }
        else{
            return rec(originalString.substring(0, originalString.length()-1),
                    reversedString + originalString.substring(originalString.length()-1));
        }
    }

    public static String reverseStringIterative(String str){
        String reversedString = "";
        for (int i=str.length()-1; i >= 0; i--){
            reversedString = reversedString + str.charAt(i);
        }
        return reversedString;
    }

    public static void main(String[] args){
        System.out.println(factorialRecursive(4));
        System.out.println(factorialIterative(4));
        System.out.println(fibonacciIterative(6));
        System.out.println(fibonacciRecursive(6));
        System.out.println(reverseStringRecursive("Apple Tree"));
        System.out.println(reverseStringIterative("Apple Tree"));
    }
}
