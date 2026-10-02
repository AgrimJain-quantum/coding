import java.util.*;

public class casec {

    static int N, M;
    static List<Integer>[] graph;

    static int start1, start2, destination;

    // pathMasks1[mask] = true if there is a valid simple path
    // from start1 to destination using exactly these towns.
    static boolean[] pathMasks1;

    // Same for scout 2
    static boolean[] pathMasks2;

    static int destinationBit;

    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        N = sc.nextInt();
        M = sc.nextInt();

        graph = new ArrayList[N];

        for (int i = 0; i < N; i++) {
            graph[i] = new ArrayList<>();
        }

        // Read roads
        for (int i = 0; i < M; i++) {

            int a = sc.nextInt() - 1;
            int b = sc.nextInt() - 1;

            graph[a].add(b);
            graph[b].add(a);
        }

        // Starting towns
        start1 = sc.nextInt() - 1;
        start2 = sc.nextInt() - 1;

        // Destination
        destination = sc.nextInt() - 1;

        int totalMasks = 1 << N;

        pathMasks1 = new boolean[totalMasks];
        pathMasks2 = new boolean[totalMasks];

        destinationBit = 1 << destination;

        // Find all possible simple paths for scout 1
        dfs(start1, 1 << start1, pathMasks1);

        // Find all possible simple paths for scout 2
        dfs(start2, 1 << start2, pathMasks2);

        int answer = Integer.MAX_VALUE;

        /*
         * Try every possible path of scout 1.
         */
        for (int mask1 = 0; mask1 < totalMasks; mask1++) {

            if (!pathMasks1[mask1]) {
                continue;
            }

            /*
             * Scout 2 cannot use any town used by scout 1,
             * except the destination.
             */
            int blocked = mask1 & ~destinationBit;

            /*
             * All towns available for scout 2.
             */
            int available = (totalMasks - 1) & ~blocked;

            /*
             * Enumerate all subsets of available towns.
             *
             * Any valid path of scout 2 must be one of these
             * subsets.
             */
            int subset = available;

            while (subset != 0) {

                if ((subset & destinationBit) != 0 &&
                    pathMasks2[subset]) {

                    int total = Integer.bitCount(mask1 | subset);

                    answer = Math.min(answer, total);
                }

                subset = (subset - 1) & available;
            }

            /*
             * Handle the case where the second path has mask 0.
             * Normally this won't happen because destination must
             * be included.
             */
        }

        if (answer == Integer.MAX_VALUE) {
            System.out.println("Impossible");
        } else {
            System.out.println(answer);
        }
    }

    /*
     * DFS to generate all simple paths from the current node
     * to the destination.
     *
     * mask = towns already visited in this path.
     */
    static void dfs(int current, int mask, boolean[] pathMasks) {

        // Destination reached
        if (current == destination) {
            pathMasks[mask] = true;
            return;
        }

        for (int next : graph[current]) {

            int nextBit = 1 << next;

            // Don't visit the same town twice
            if ((mask & nextBit) != 0) {
                continue;
            }

            dfs(next, mask | nextBit, pathMasks);
        }
    }
}