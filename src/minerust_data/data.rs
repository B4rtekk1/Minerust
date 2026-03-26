use directories::ProjectDirs;

pub fn get_project_dirs() -> Result<ProjectDirs, &'static str> {
    ProjectDirs::from("com", "Minerust", "Minerust")
        .ok_or("Could not determinate directory (ProjectDirs)")
}
