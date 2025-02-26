import java.io.*; // Import for file handling
import java.util.*; // Import for List, ArrayList, and Scanner

// This class manages student records, including adding, displaying, searching, and saving students to a file
public class StudentManager {
    private List<Student> students; // List to store students in memory
    private static final String FILE_NAME = "students.txt"; // File to store students persistently

    // Constructor - initializes the student list and loads data from file (if
    // exists)
    public StudentManager() {
        this.students = new ArrayList<>(); // Create an empty list to store students
        loadStudentsFromFile(); // Load existing student records from file when program starts
    }

    // ✅ Method to add a new student to the list and save to file
    public void addStudent(String name, int age, String id) {
        students.add(new Student(name, age, id)); // Create a new student object and add it to the list
        saveStudentsToFile(); // Save the updated student list to the file
    }

    // ✅ Method to display all students
    public void displayStudents() {
        if (students.isEmpty()) { // If the list is empty, inform the user
            System.out.println("No students found.");
            return;
        }
        // Print details of each student
        for (Student student : students) {
            System.out.println(student);
        }
    }

    // ✅ Method to search for a student by their ID
    public Student searchStudent(String id) {
        for (Student student : students) { // Loop through all students
            if (student.getId().equals(id)) { // If the ID matches, return the student object
                return student;
            }
        }
        return null; // Return null if no student is found
    }

    // ✅ Method to save all students to a file (Persistence)
    private void saveStudentsToFile() {
        try (PrintWriter writer = new PrintWriter(new FileWriter(FILE_NAME))) {
            for (Student student : students) {
                // Save student data in the format: ID,Name,Age
                writer.println(student.getId() + "," + student.getName() + "," + student.getAge());
            }
        } catch (IOException e) {
            System.out.println("Error saving students to file: " + e.getMessage());
        }
    }

    // ✅ Method to load student records from a file (Executed when the program
    // starts)
    private void loadStudentsFromFile() {
        File file = new File(FILE_NAME); // Create a File object for students.txt
        if (!file.exists())
            return; // If the file does not exist, do nothing and return

        try (Scanner scanner = new Scanner(file)) { // Scanner reads the file line by line
            while (scanner.hasNextLine()) { // While there are lines in the file
                String[] data = scanner.nextLine().split(","); // Split the line by comma into an array
                // Create a Student object and add it to the list
                students.add(new Student(data[1], Integer.parseInt(data[2]), data[0]));
            }
        } catch (IOException e) {
            System.out.println("Error loading students from file: " + e.getMessage());
        }
    }
}
