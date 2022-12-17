const { Transaction,SousAgence,Agence, Balance } = require('../../db/sequelize')

  
module.exports = (app) => {
  app.post('/api/InsertTransaction', (req, res) => {

  
   Transaction.create({
    montant_a_recevoir:+req.body.montant_a_recevoir,
    montantTotal:+req.body.montantTotal,
    status:req.body.status,
    paysDest:req.body.paysDest,
    frais:+req.body.frais,
    paysOrigine:req.body.paysOrigine,
    DeviceDest:req.body.DeviceDest,
    DeviceOrigine:req.body.DeviceOrigine,
    DEVISEId:+req.body.DEVISEId,
    CLIENTId:+req.body.CLIENTId,
    UserId:+req.body.UserId,
    recepteurid:+req.body.recepteurid
    
  

    })
    .then(result=>{
      SousAgence.findOne({where:{id:+req.body.idsousagence}})
      .then(resultat1=>{
           console.log("suis la l_id agence "+ resultat1.AGENCEId)
           Balance.findOne({where:{AGENCEId:resultat1.AGENCEId}})
           .then(resultatfinal=>{
            console.log("suis la montant"+resultatfinal.montant);
            if(req.body.montantTotal <resultatfinal.montant){
              montantupdate=resultatfinal.montant-req.body.montantTotal
              resultatfinal.update({montant:montantupdate})
              console.log("suis la dans transaction"+req.body.idsousagence)
              message='insertion passé avec succés'
              erreur='false'
              res.json({message,erreur,data:result.id,erreur})
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
      })
     
    })
    .catch(error=>{
      message='veuillez vérifier vos informations de transaction'
      res.json({message})
    })
    
    
   
   
  })
}