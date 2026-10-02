import java.util.*;
import java.util.regex.*;

public class caseb {

    static class Brick {
        char type;

        Brick(char type) {
            this.type = type;
        }
    }

    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        int n = sc.nextInt();
        sc.nextLine();

        List<Brick> bricks = new ArrayList<>();

        // -1 means empty
        int[][] grid = new int[n][100];

        for (int[] row : grid) {
            Arrays.fill(row, -1);
        }

        int start = -1;
        int destination = -1;

        Pattern pattern = Pattern.compile("(\\d+)([RGSD])");

        // -----------------------------
        // Read and parse the wall
        // -----------------------------
        for (int r = 0; r < n; r++) {

            String line = sc.nextLine().trim();

            Matcher matcher = pattern.matcher(line);

            int col = 0;

            while (matcher.find()) {

                int length = Integer.parseInt(matcher.group(1));
                char type = matcher.group(2).charAt(0);

                // This represents ONE brick
                int id = bricks.size();

                bricks.add(new Brick(type));

                // Fill all cells occupied by this brick
                for (int k = 0; k < length; k++) {
                    grid[r][col] = id;
                    col++;
                }

                if (type == 'S') {
                    start = id;
                }

                if (type == 'D') {
                    destination = id;
                }
            }
        }

        int totalBricks = bricks.size();

        // -----------------------------
        // Build graph
        // -----------------------------
        List<Set<Integer>> graph = new ArrayList<>();

        for (int i = 0; i < totalBricks; i++) {
            graph.add(new HashSet<>());
        }

        int[] dr = {-1, 1, 0, 0};
        int[] dc = {0, 0, -1, 1};

        for (int r = 0; r < n; r++) {

            for (int c = 0; c < 100; c++) {

                int current = grid[r][c];

                if (current == -1) {
                    continue;
                }

                for (int d = 0; d < 4; d++) {

                    int nr = r + dr[d];
                    int nc = c + dc[d];

                    if (nr < 0 || nr >= n ||
                        nc < 0 || nc >= 100) {
                        continue;
                    }

                    int next = grid[nr][nc];

                    if (next == -1 || next == current) {
                        continue;
                    }

                    graph.get(current).add(next);
                    graph.get(next).add(current);
                }
            }
        }

        // -----------------------------
        // 0-1 BFS
        // -----------------------------

        int INF = Integer.MAX_VALUE;

        int[] dist = new int[totalBricks];

        Arrays.fill(dist, INF);

        Deque<Integer> deque = new ArrayDeque<>();

        dist[start] = 0;
        deque.addFirst(start);

        while (!deque.isEmpty()) {

            int current = deque.pollFirst();

            if (current == destination) {
                break;
            }

            for (int next : graph.get(current)) {

                // Red bricks cannot be broken/used
                if (bricks.get(next).type == 'R') {
                    continue;
                }

                /*
                 * Green brick costs 1 to break.
                 * S and D cost 0.
                 */
                int cost = 0;

                if (bricks.get(next).type == 'G') {
                    cost = 1;
                }

                int newDistance = dist[current] + cost;

                if (newDistance < dist[next]) {

                    dist[next] = newDistance;

                    if (cost == 0) {
                        deque.addFirst(next);
                    } else {
                        deque.addLast(next);
                    }
                }
            }
        }

        System.out.println(dist[destination]);
    }
}