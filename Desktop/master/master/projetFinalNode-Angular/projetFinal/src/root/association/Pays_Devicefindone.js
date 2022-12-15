
const { Pays, Devise } = require('../../db/sequelize')

  
module.exports = (app) => {
  app.post('/api/findonePays_Devices', (req, res) => {
  
  
      Pays.findOne({where:{id:+req.body.id}})
      .then(resultat => {
        const message = `La liste d'un pays et sa devise a bien été récupérée.`

        
       
        Devise.findOne({id:resultat.DEVISEId})
        .then(resultat1=>{
          res.json({ message, data:resultat,resultat1})
        })
      })
     .catch(error=>{
      const message="la liste de ce  Pays avec sa devise n_est pas disponible n'est pas disponibles.Réssayer dans quelques instant!"
      res.status(500).json({message,data:error})
     })
    
   
  })
}