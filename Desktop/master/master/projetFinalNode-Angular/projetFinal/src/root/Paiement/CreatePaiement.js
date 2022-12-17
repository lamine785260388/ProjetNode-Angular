
const { Client, Paiement, Transaction,SousAgence,Balance } = require('../../db/sequelize')


  
module.exports = (app) => {
  app.post('/api/InsertPaiement', (req, res) => {
  
  
      Transaction.findByPk(req.body.id)
      .then(result => {
        console.log(result.status)
        if(result!=null){
          if(result.status=='envoye' && result.recepteurid==req.body.recepteurid){
            Paiement.create({
             
              numero_piece:req.body.recepteurid,
              nom_recepteur:req.body.nom_recepteur,
              TRANSACTIONId:+req.body.id,
            })
            result.update({status:'payé'})
            SousAgence.findByPk(req.body.idsousAgence)
            .then(resultatsousAgence=>{
                 console.log(resultatsousAgence.AGENCEId);
                 Balance.findOne({where:{AGENCEId:resultatsousAgence.AGENCEId}})
                 .then(resultatfinal=>{
                  console.log("suis la montant"+resultatfinal.montant);
                  console.log(result.montant_a_recevoir)
                  if(result.montantTotal <resultatfinal.montant){
                    montantupdate=resultatfinal.montant+result.montant_a_recevoir
                    resultatfinal.update({montant:montantupdate})
                    .then(result=>{
                      res.json('success')
                    })
                     .catch(erreur=>{
                      console.log(erreur);
                     })
                   // console.log("suis la dans transaction"+req.body.idsousagence)
                    
                   
                   
                  }
                  else{
                    Transaction.destroy({
                      where: { id: result.id }
                    })
                    console.log('suis la dans le else');
                    message='Votre montant est insuffisante pour faire cette Transaction votre solde est '+resultatfinal.montant
                    res.json({message})
      
                  }
                 })
                 .catch(erreur=>{
                    console.log(erreur);
                 })
            })
           
          const message = 'La liste de Transaction a bien été récupérée.'
          const erreur="false"
           res.json({ message,erreur, data: result,erreur })}
          else{
            return res.json('vous n_avez de ')
          }
        }
        else{
          const message1 = 'La liste de paiement'
          
          res.json({message1})
        }
        
      })
     .catch(error=>{
      const message="la liste des Transactions n'est pas disponibles.Résseyer dans quelques instant!"
      res.status(200).json({message,data:error})
     })
    
   
  })
}
//InsertPaiement