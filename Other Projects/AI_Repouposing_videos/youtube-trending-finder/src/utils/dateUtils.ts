export function getCurrentDate(): string {
    return new Date().toISOString();
}

export function formatDateForApi(date: Date): string {
    return date.toISOString().split('T')[0]; // Returns date in YYYY-MM-DD format
}

export function isWithinLast24Hours(date: Date): boolean {
    const now = new Date();
    const hoursDifference = (now.getTime() - date.getTime()) / (1000 * 60 * 60);
    return hoursDifference <= 24;
}