
const { Paiement } = require('../../db/sequelize')

  
module.exports = (app) => {
  app.get('/api/findAllPaiement', (req, res) => {
  
  
      Paiement.findAll({order:['nom_recepteur' ]})
      .then(agence => {
        const message = 'La liste des Paiements a bien été récupérée.'
        res.json({ message, data: agence })
      })
     .catch(error=>{
      const message="la liste des Paiements n'est pas disponible.Résseyer dans quelques instant!"
      res.status(500).json({message,data:error})
     })
    
   
  })
}