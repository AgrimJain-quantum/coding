import java.util.*;

public class test2 {

    static class Command {
        int existing;
        int newCube;
        String direction;

        Command(int existing, int newCube, String direction) {
            this.existing = existing;
            this.newCube = newCube;
            this.direction = direction;
        }
    }

    static class Position {
        int x, y;

        Position(int x, int y) {
            this.x = x;
            this.y = y;
        }
    }

    static String key(int x, int y) {
        return x + "," + y;
    }

    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        int N = sc.nextInt();

        List<Command> commands = new ArrayList<>();

        for (int i = 0; i < N; i++) {
            int existing = sc.nextInt();
            int newCube = sc.nextInt();
            String direction = sc.next();

            commands.add(new Command(existing, newCube, direction));
        }

        int target = sc.nextInt();

        // Sort by existing cube, then by new cube
        commands.sort((a, b) -> {
            if (a.existing != b.existing) {
                return Integer.compare(a.existing, b.existing);
            }
            return Integer.compare(a.newCube, b.newCube);
        });

        // cube -> position
        Map<Integer, Position> cubePos = new HashMap<>();

        // position -> cube
        Map<String, Integer> grid = new HashMap<>();

        /*
         * Cube 1 is the starting cube.
         * Put it at (0, 0).
         */
        cubePos.put(1, new Position(0, 0));
        grid.put(key(0, 0), 1);

        for (Command cmd : commands) {

            Position p = cubePos.get(cmd.existing);

            int x = p.x;
            int y = p.y;

            // Find position of new cube
            if (cmd.direction.equals("top")) {
                y++;
            }
            else if (cmd.direction.equals("down")) {
                y--;
            }
            else if (cmd.direction.equals("left")) {
                x--;
            }
            else if (cmd.direction.equals("right")) {
                x++;
            }

            String newKey = key(x, y);

            /*
             * If a cube already exists at this position,
             * it gets replaced.
             */
            if (grid.containsKey(newKey)) {
                int oldCube = grid.get(newKey);
                cubePos.remove(oldCube);
            }

            // Place new cube
            cubePos.put(cmd.newCube, new Position(x, y));
            grid.put(newKey, cmd.newCube);
        }

        // Position of target cube
        Position p = cubePos.get(target);

        int up = grid.getOrDefault(key(p.x, p.y + 1), -1);
        int down = grid.getOrDefault(key(p.x, p.y - 1), -1);
        int left = grid.getOrDefault(key(p.x - 1, p.y), -1);
        int right = grid.getOrDefault(key(p.x + 1, p.y), -1);

        System.out.println(up + " " + down + " " + left + " " + right);
    }
}