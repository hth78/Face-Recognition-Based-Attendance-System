# Face-Recognition-Based-Attendance-System
<a href="https://www.youtube.com/watch?v=dQw4w9WgXcQ"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"></a>
<p align="center">
  <img src="https://user-images.githubusercontent.com/58062535/175368840-ae4c10a7-d6c1-4666-a218-656181eb80dc.png" />
</p>
<a href="https://www.youtube.com/watch?v=dQw4w9WgXcQ"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"></a>

## Face Recognition
Face recognition is a biometric recognition technique. Biometric recognition is an information system that allows the identification of a person based on some of its main physiological and behavioral characteristics. Face recognition is a broad problem of identifying or verifying people in photographs and videos, a process comprised of detection, alignment, feature extraction, and a recognition task It has 4 steps which are :

1. Face Detection
2. Data Gathering
3. Data Comparison
4. Face Recognition

## About the project
This is a `python-based`application which provides an involuntary attendance marking system that operates without human intervention. It intends to serve as an efficient substitute for traditional manual attendance systems. It can be used in corporate offices, schools, and organizations where security is essential.

The system can also be used to maintain track of employees or students or other instructional activities where attendance is critical. In order to be recognised, students or employees must also register in the database. The user-friendly interface allows for on-the-spot registration. Using an admin account and password, the organization's admin can email the attendance of the day to the respective email address. Only the organization's administrators will have access to the attendance information, and no changes will be made to the attendance details.

It aims to automate the traditional attendance system where the attendance is marked manually. It also enables an organization to maintain its records like in-time, out time, break time and attendance digitally. Digitalization of the system would also help in better visualization of the data using graphs to display the no. of employees present today, total work hours of each employee and their break time. Its added features serve as an efficient upgrade and replacement over the traditional attendance system.

## Scope of the project
Facial recognition is becoming more prominent in our society. It has made major progress in the field of security. It is a very effective tool that can help low enforcers to recognize criminals and software companies are leveraging the technology to help users access the technology. This technology can be further developed to be used in other avenues such as ATMs, accessing confidential files, or other sensitive materials. This project servers as a foundation for future projects based on facial detection and recognition. This project also converts web development and database management with a user-friendly UI. Using this system any corporate offices, school and organization can replace their traditional way of maintaining attendance of the employees and can also generate their availability(presence) report throughout the month.

## Tech Stack Used
### Built with
* Jupyter Notebook

### Modules Used
* OpenCV
* tkinter
* Numpy
* Pandas
* imaplib
* smtplib
* yagmail
* CSV
### Facial Recognition Algorithms
* SCRFD / RetinaFace-style detection via InsightFace
* ArcFace embeddings with cosine-similarity matching
* Legacy LBPH/Haar kept only as an optional compatibility adapter

## Flowchart of the project
<p align="center">
  <img src="https://user-images.githubusercontent.com/58062535/175374824-1d2e58a7-403f-4fdd-858e-61bc970ed1b1.png" />
</p>

## GUI of the project

The snapshots below demonstrate the GUI Interface of the attendance system, which is divided into several sections as mentioned below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/58062535/176115084-2ede7a80-c3d3-4787-a4a3-f8e6e90d232d.png" />
</p>
Main Window: It contains the options such as Admin, Member and Quit. The user will choose the preferred option based on its accessibility.
<p align="center">
  <img src="https://user-images.githubusercontent.com/58062535/176115357-9daa4860-bcbf-4596-ac9c-181ce8b47fc2.png" />
</p>
Admin Login Window: It is used to input the username and password of the admin account. If the username and password match with the admin account then it will login to the next window otherwise it will prompt you to re-enter the username and password.
<p align="center">
  <img src="https://user-images.githubusercontent.com/58062535/176115530-7350c70f-3238-49e4-804a-f5ccb3745ece.png" />
</p>
AutoMail Window: This window allows the admin to mail the attendance report to the respective mail ID from the admin’s account credentials. This window will login only when the user will be able to provide the correct admin username and password.
<p align="center">
  <img src="https://user-images.githubusercontent.com/58062535/176115710-02b34564-01db-465f-abd6-2c162aec4c11.png"/>
</p>


## Secure authentication setup
This project no longer stores SMTP/IMAP or admin credentials in code. Configure secrets before running notebook cells that initialize email/authentication.

### Required environment variables
- `FRAS_EMAIL_USERNAME`: SMTP/IMAP username (for example, your mailbox address).
- `FRAS_EMAIL_PASSWORD`: SMTP/IMAP password or app password.
- `FRAS_ADMIN_USERNAME`: Username required by the admin login window.
- `FRAS_ADMIN_PASSWORD`: Password required by the admin login window.

### Optional mail host overrides
- `FRAS_IMAP_HOST` (default: `imap.gmail.com`)
- `FRAS_IMAP_PORT` (default: `993`)
- `FRAS_SMTP_HOST` (default: `smtp.gmail.com`)
- `FRAS_SMTP_PORT` (default: `587`)

### Quick setup example
```bash
export FRAS_EMAIL_USERNAME="your-mailbox@example.com"
export FRAS_EMAIL_PASSWORD="your-app-password"
export FRAS_ADMIN_USERNAME="your-admin-username"
export FRAS_ADMIN_PASSWORD="strong-admin-password"
```

You can also integrate a secret manager by passing a provider callback to `security_config.load_email_credentials()` / `load_admin_credentials()`.

### Credential rotation guidance
Credentials that were previously committed to source control must be treated as compromised. Rotate/revoke them in your mail provider immediately, then update your local environment values.

## Team Members
[Riya Negi](https://github.com/riyanegi1211)

[Mohak Kala](https://github.com/MohakKala07)


## Python package layout

Core notebook logic has been modularized into `attendance_system/`:

- `attendance_system/vision/detector.py`: modern face detection wrapper (InsightFace/SCRFD).
- `attendance_system/vision/embedder.py`: modern face embedding wrapper (ArcFace via InsightFace).
- `attendance_system/vision/matcher.py`: cosine-similarity matcher with configurable threshold.
- `attendance_system/legacy/lbph_adapter.py`: rollback-only compatibility adapter for LBPH/Haar.
- `config.yaml`: detector/embedder/matcher configuration (model names, detector size, threshold).
