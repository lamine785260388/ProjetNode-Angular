export class Paiement {
    constructor(
        protected date_paiement : Date,
        protected numero_piece : string,
        protected nom_recepteur : string
    )
    {}
}
