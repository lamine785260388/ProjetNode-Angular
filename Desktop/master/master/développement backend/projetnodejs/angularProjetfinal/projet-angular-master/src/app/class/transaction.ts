export class Transaction {
    constructor(
        _montant_a_recevoir : GLfloat,
        public date_envoie : Date,
        public frais : GLfloat, 
        public statut_transaction : string,
        public pays_origne : string,
        public pays_destination : string, 
        public devise_origine : string,
        public devise_destinataire : string

        
    ){}
}
