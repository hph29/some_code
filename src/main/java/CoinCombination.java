import java.util.Arrays;

public class CoinCombination {

    int listNumCombination(int[] coins, int sum){
      if (sum == 0){
          return 1;
      }
      else if (sum < 0 || coins.length == 0){
          return 0;
      }
      else {
          return listNumCombination(Arrays.copyOf(coins, coins.length-1), sum) +
                  listNumCombination(coins, sum - coins[coins.length - 1]);
      }

    }
}
