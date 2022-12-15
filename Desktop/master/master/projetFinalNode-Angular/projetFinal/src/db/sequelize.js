const { Sequelize, DataTypes } = require('sequelize')
const UserModel=require('../models/user')
const bcrypt=require('bcrypt')
const TransactionModel=require('../models/transaction')
const PaiementModel=require('../models/paiement')
const ClientModel=require('../models/client')
const DeviseModel=require('../models/devise')
const paysModel=require('../models/Pays')
const agenceModel=require('../models/agence')
const SousAgenceModel=require('../models/SousAgence')
const BalanceModel=require('../models/balance')
const VilleModel=require('../models/ville')

const sequelize = new Sequelize('nodejsangular', 'root', '', {
  host: 'localhost',
  dialect: 'mariadb',
  dialectOptions: {
    timezone: 'Etc/GMT-2',
  },
  logging: false
})
  
const User=UserModel(sequelize,DataTypes)
const Transaction=TransactionModel(sequelize,DataTypes)
const Paiement=PaiementModel(sequelize,DataTypes)
const Client=ClientModel(sequelize,DataTypes)
  const Devise=DeviseModel(sequelize,DataTypes)
  const Pays=paysModel(sequelize,DataTypes)
  const Agence=agenceModel(sequelize,DataTypes)
  const SousAgence=SousAgenceModel(sequelize,DataTypes)
  const Balance=BalanceModel(sequelize,DataTypes)
  const Ville=VilleModel(sequelize,DataTypes)
const initDb = () => {
  return sequelize.sync({force:false}).then(_ => {
  
    // bcrypt.hash('lamine',10)
    // .then(hash=>User.create({
    //   username:'lamine',
    //   isAdmin:'true',
     
    //   password:hash
    // })
    // .then(user=>console.log(user.toJSON()))

    // )
   
    console.log('La base de donnée a bien été synchronisé  !')
  })
}
//relation entre paiement et transaction
// Transaction.hasOne(Paiement);
// Paiement.belongsTo(Transaction);
Transaction.hasOne(Paiement, {
  foreignKey: 'TRANSACTIONId'
});
Paiement.belongsTo(Transaction);

//relation entre transaction et devise
Devise.hasMany(Transaction);
Transaction.belongsTo(Devise);
//relation entre client et transaction
Client.hasMany(Transaction);
Transaction.belongsTo(Client);
//relation user et transaction
User.hasMany(Transaction);
Transaction.belongsTo(User);
//relation entre devise et pays
Devise.hasMany(Pays);
Pays.belongsTo(Devise);
//relation entre pays et ville
Pays.hasMany(Ville);
Ville.belongsTo(Pays);
//relation entre user et sousAgence
SousAgence.hasMany(User);
 User.belongsTo(SousAgence);
//relation entre sous agence et Agence
  Agence.hasMany(SousAgence);
  SousAgence.belongsTo(Agence);
//relation entre Agence et Balance
Agence.hasOne(Balance);
Balance.belongsTo(Agence);
module.exports = { 
  initDb,User,Transaction,Paiement,Client,Devise,Agence,Pays,SousAgence,Balance,Ville
}