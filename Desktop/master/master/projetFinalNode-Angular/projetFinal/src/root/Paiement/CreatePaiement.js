const { Paiement } = require('../../db/sequelize')

  
module.exports = (app) => {
  app.post('/api/InsertPaiement', (req, res) => {

  
   Paiement.create({
    date_paiement:req.body.date_paiement,
    numero_piece:req.body.numero_piece,
    nom_recepteur:req.body.nom_recepteur,
    TRANSACTIONId:+req.body.TRANSACTIONId,
  

    })
    return res.json('insertion Paiement success')
    
   
   
  })
}