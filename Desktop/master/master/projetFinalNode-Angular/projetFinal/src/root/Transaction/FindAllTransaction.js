
const { Transaction } = require('../../db/sequelize')

  
module.exports = (app) => {
  app.get('/api/findAllTransaction', (req, res) => {
  
  
      Transaction.findAll({order:['createdAt' ]})
      .then(trans => {
        const message = 'La liste des Transaction a bien été récupérée.'
        res.json({ message, data: trans })
      })
     .catch(error=>{
      const message="la liste des Transactions n'est pas disponible.Résseyer dans quelques instant!"
      res.status(500).json({message,data:error})
     })
    
   
  })
}