class Video {
    id: string;
    title: string;
    description: string;
    viewCount: number;

    constructor(id: string, title: string, description: string, viewCount: number) {
        this.id = id;
        this.title = title;
        this.description = description;
        this.viewCount = viewCount;
    }

    formatVideoInfo(): string {
        return `${this.title} (ID: ${this.id}) - ${this.viewCount} views\nDescription: ${this.description}`;
    }
}

export default Video;