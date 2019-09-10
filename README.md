# Human Activity Recognition System using Machine Learning techniques for Home Automation

Thesis for my BSc degree in Computer Science at Aristotle University

## Jupyter Notebook / License
Code is publicly available [here](https://github.com/LeonVitanos/Thesis/blob/master/Thesis.ipynb). See the [LICENSE](https://github.com/LeonVitanos/Thesis/blob/master/LICENSE) file for license rights and limitations (MIT).

## Abstract

Human activity recognition (HAR), as a branch of human-computer interaction and pervasive computing, has been extensively investigated over the last two decades and remarkable advances have been made. Today, most mobile devices have fast processors with low power consumption and built-in accurate compact sensors. Data from sensors such as the accelerometer allow people to construct an acceleration-based HAR system through the training of machine learning models without the need for additional hardware. Driven by the immense success of deep learning, the research paradigm has been shifted from traditional approaches to deep learning methods which have significantly pushed the state-of-the-art in human activity recognition. In contrast to statistical methods, deep learning makes it possible to extract high-level features automatically, thereby achieving promising performance for the recognition of human activities through the use of sensor-based HAR systems.

The scope of this thesis is to design and implement a user-independent human activity recognition system with possible applications in assisted living technologies. The aim of the system is to interpret human body gestures through the acceleration data of a tri-axial accelerometer, which is integrated with most modern mobile devices. To solve this optimization problem, a heuristic approach is proposed, to achieve an optimal solution with low computational cost. The proposed approach, is the use of the Haar wavelet transform for data smoothing, and to train a deep neural network without any feature extraction method.

In this paper, we used a dataset where eight users performed twenty repetitions of twenty different gestures to collect 3251 sequences. From an initial set of seven classifiers, the one with the best accuracy was selected. The proposed system is able to classify gestures, executed at different speeds, with minimal pre-processing making it computationally efficient.

Finally, an Android application was developed that contained the model with the best accuracy. The application communicates to a Raspberry Pi after recognizing the gesture that the user has performed and executes the corresponding script with the gesture recognized. Possible actions would be to perform daily activities related to home automation to support older or disabled people.



## Περίληψη

Η αναγνώριση ανθρώπινης δραστηριότητας (HAR), ως κλάδος της αλληλεπίδρασης ανθρώπου-υπολογιστή και της διάχυτης πληροφορικής, έχει διερευνηθεί εκτενώς τις τελευταίες δύο δεκαετίες και έχουν επιτευχθεί αξιοσημείωτες πρόοδοι. Σήμερα, οι περισσότερες κινητές συσκευές έχουν γρήγορους επεξεργαστές με χαμηλή κατανάλωση ενέργειας και ενσωματωμένους αισθητήρες μικρού μεγέθους με μεγάλη ακρίβεια. Tα δεδομένα από αισθητήρες όπως το επιταχυνσιόμετρο επιτρέπουν στους ανθρώπους να κατασκευάσουν ένα σύστημα HAR μέσω της εκπαίδευσης μοντέλων μηχανικής μάθησης χωρίς την ανάγκη πρόσθετου εξοπλισμού. Πρόσφατα, οι μέθοδοι βαθιάς εκμάθησης έχουν ωθήσει σημαντικά την τελευταία λέξη της τεχνολογίας στην αναγνώριση της ανθρώπινης δραστηριότητας. Σε αντίθεση με τις στατιστικές μεθόδους, η βαθιά μάθηση καθιστά δυνατή την αυτόματη εξαγωγή χαρακτηριστικών υψηλού επιπέδου, επιτυγχάνοντας έτσι πολλά υποσχόμενη απόδοση για την αναγνώριση ανθρώπινων δραστηριοτήτων μέσω συστημάτων HAR που βασίζονται σε αισθητήρες.

Σκοπός της παρούσας εργασίας είναι ο σχεδιασμός και η υλοποίηση ενός συστήματος αναγνώρισης ανθρώπινης δραστηριότητας, το οποίο θα είναι ανεξάρτητο από τον χρήστη, με πιθανές εφαρμογές σε τεχνολογίες υποβοηθούμενης διαβίωσης. Στόχος του συστήματος είναι η ερμήνευση χειρονομιών του ανθρώπινου σώματος μέσω των δεδομένων επιτάχυνσης ενός τρι-αξονικού επιταχυνσιόμετρου, ο οποίος είναι ενσωματωμένος στις περισσότερες σύγχρονες κινητές συσκευές. Για την επίλυση του προβλήματος βελτιστοποίησης της ακρίβειας του συστήματος προτείνεται μια ευρετική προσέγγιση, ώστε να επιτευχθεί βέλτιστη λύση με χαμηλό υπολογιστικό κόστος. Η μέθοδος επίλυσης που επιλέχτηκε είναι η εξομάλυνση των δεδομένων μέσω του μετασχηματισμού Haar και η εκπαίδευση ενός βαθιού νευρωνικού δικτύου χωρίς χρήση κάποιας μεθόδου εξαγωγής χαρακτηριστικών. 

Στην παρούσα εργασία, χρησιμοποιήθηκε μια συλλογή δεδομένων, όπου οκτώ χρήστες εκτέλεσαν είκοσι επαναλήψεις από είκοσι διαφορετικές χειρονομίες για την συλλογή 3251 αλληλουχιών. Από ένα αρχικό σύνολο επτά κατηγοριοποιητών επιλέχθηκε αυτός με την καλύτερη ακρίβεια. Το προτεινόμενο σύστημα είναι σε θέση να κατηγοριοποιεί χειρονομίες, που εκτελούνται με διαφορετικές ταχύτητες, με ελάχιστη προεπεξεργασία καθιστώντας το υπολογιστικά αποδοτικό.

Τέλος, αναπτύχθηκε μια εφαρμογή Android, η οποία περιείχε το μοντέλο με την μεγαλύτερη ακρίβεια. Η εφαρμογή αφού αναγνωρίσει την χειρονομία που εκτελέσει ο χρήστης, επικοινωνεί με ένα Raspberry Pi, το οποίο εκτελεί το αντίστοιχο αρχείο μικροεντολών με την χειρονομία που αναγνωρίστηκε. Πιθανές ενέργειες θα ήταν η εκτέλεση καθημερινών δραστηριοτήτων που σχετίζονται με τον οικιακό αυτοματισμό,  για την υποστήριξη ηλικιωμένων ή ατόμων με ειδικές ανάγκες. 
